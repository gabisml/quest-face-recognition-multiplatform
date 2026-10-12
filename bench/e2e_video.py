"""Benchmark 2 — end-to-end video.

Runs the full FaceRecognizer pipeline frame-by-frame on each video. After
a 60s warm-up window, records per-frame latency. Computes:

    - fps_mean (over the steady-state window)
    - latency_median_ms, latency_p95_ms, latency_max_ms
    - frac_in_33ms, frac_in_66ms, frac_in_100ms (FRUITS-style budgets)

System monitor samples CPU/RAM/GPU/etc. on a background thread for the
duration. Throttling on RPi triggers a noisy abort.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from bench.config import (
    LATENCY_BUDGETS_MS,
    MIN_VIDEO_DURATION_S,
    VIDEO_WARMUP_S,
    result_dir,
)
from bench.hardware import build_system_info
from bench.quality import MARGEM_PADRAO, Avaliador, carregar_groundtruth
from bench.sync import cuda_sync
from bench.system_monitor import SystemMonitor, reset_torch_peak_memory

LOG = logging.getLogger("bench.e2e_video")


def _frame_iter(cap: cv2.VideoCapture) -> Iterable[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        yield frame


def _open_video(video_path: Path):
    """Open the video with imageio (ffmpeg embutido em imageio-ffmpeg).

    cv2.VideoCapture varia por hardware — alguns builds locais foram compilados
    sem FFmpeg, outros só com GStreamer (que falha em seekar MP4). imageio dá
    comportamento idêntico em todos os hardwares e fica fora da janela de
    medição (read_frame() é chamado entre cuda_sync/perf_counter_ns).

    Returns (read_fn, reset_fn, release_fn, backend_name).
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot open video: {video_path} — imageio[ffmpeg] não instalado. "
            f"Rode `pip install imageio imageio-ffmpeg`."
        ) from exc

    # Sanity: garantir que o arquivo existe e é decodificável.
    try:
        _probe = next(iio.imiter(str(video_path)))
        del _probe
    except Exception as exc:
        raise RuntimeError(f"Cannot open video: {video_path}") from exc

    state = {"iter": iio.imiter(str(video_path))}

    def read():
        try:
            frame_rgb = next(state["iter"])
        except StopIteration:
            return None
        return frame_rgb[:, :, ::-1].copy()  # RGB → BGR

    def reset():
        state["iter"] = iio.imiter(str(video_path))

    def release():
        pass

    return read, reset, release, "imageio"


# Quadros do aquecimento, mantidos em memoria. 800x600x3 dao ~1,4 MB por quadro,
# entao 120 quadros custam ~173 MB -- cabe ate no Orin Nano.
WARMUP_BUFFER_FRAMES = 120


def _preload_warmup_frames(video_path: Path, n: int) -> list[np.ndarray]:
    """Le os primeiros n quadros para a memoria, ANTES de qualquer trabalho CUDA.

    Existe por causa de um defeito de plataforma nos Jetson (Tegra): criar um
    processo filho depois que o backend CUDA do DNN do OpenCV inicializou seu
    estado de GPU corrompe silenciosamente as deteccoes seguintes -- o detector
    passa a inventar ou perder rostos, de forma fixa dentro do processo e
    diferente entre processos. O `imageio` cria leitor gerando um subprocesso
    `ffmpeg`, e o aquecimento precisava reabrir o video a cada volta. Aquecendo
    sobre um buffer em memoria nenhum leitor nasce depois da primeira deteccao.
    Reproduzido na AGX Orin e no Orin Nano; o DGX Spark (nao-Tegra) e imune.
    """
    read_frame, _, _, _ = _open_video(video_path)
    quadros: list[np.ndarray] = []
    while len(quadros) < n:
        frame = read_frame()
        if frame is None:
            break
        quadros.append(frame)
    return quadros


def run_video(
    video_path: Path,
    recognizer,
    device: str,
    warmup_s: float,
    min_duration_s: float,
    loop_if_short: bool = True,
    emotion_classifier=None,
    avaliador: "Avaliador | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the recognizer over a video; return per-frame DataFrame + summary.

    Se `emotion_classifier` for passado, classifica emoção das faces alinhadas
    devolvidas por recognize_frame e mede o tempo da etapa separadamente.
    """
    # Ordem importa: os dois leitores nascem AQUI, antes de qualquer deteccao.
    # Ver _preload_warmup_frames -- em Tegra, criar subprocesso depois que o
    # backend CUDA do OpenCV inicializou corrompe as deteccoes seguintes.
    read_frame, reset_video, release_video, backend = _open_video(video_path)
    # iio.imiter e preguicoso: so gera o processo do ffmpeg no primeiro next().
    # Ler um quadro agora forca esse nascimento antes da primeira deteccao; o
    # quadro fica guardado e volta como o primeiro da medicao, sem perder nada.
    _primeiro = {"frame": read_frame()}

    def ler_quadro():
        if _primeiro["frame"] is not None:
            frame, _primeiro["frame"] = _primeiro["frame"], None
            return frame
        return read_frame()

    warmup_frames = _preload_warmup_frames(video_path, WARMUP_BUFFER_FRAMES) if warmup_s > 0 else []
    LOG.info("[%s] backend=%s emotion=%s | buffer de warm-up: %d quadros",
             video_path.name, backend, emotion_classifier is not None, len(warmup_frames))

    rows: list[dict] = []
    indice_quadro = 0

    # --- Aquecimento sobre o buffer em memoria, sem tocar no leitor da medicao ---
    if warmup_s > 0 and warmup_frames:
        LOG.info("[%s] warm-up %.0f s | loop=%s", video_path.name, warmup_s, loop_if_short)
        aquece_ate = time.perf_counter() + warmup_s
        i = 0
        while time.perf_counter() < aquece_ate:
            cuda_sync(device)
            recognizer.recognize_frame(warmup_frames[i % len(warmup_frames)])
            cuda_sync(device)
            i += 1

    warmup_until = time.perf_counter()
    steady_state_end = warmup_until + min_duration_s
    LOG.info("[%s] MEASUREMENT_BEGIN", video_path.name)
    in_steady = True

    while True:
        frame = ler_quadro()
        if frame is None:
            if not loop_if_short or time.perf_counter() >= steady_state_end:
                # Single-pass: fim do video encerra a medicao. Loop: encerra ao
                # cumprir a janela.
                break
            # Modo com loop: reabrir o video aqui recria o subprocesso e, em
            # Tegra, envenena as deteccoes do resto da janela. Use --no-loop
            # nesses alvos.
            reset_video()
            continue

        cuda_sync(device)
        t0 = time.perf_counter_ns()
        reconhecimentos = recognizer.recognize_frame(frame)
        emotion_ms = 0.0
        if emotion_classifier is not None:
            aligned_faces = [
                d["aligned_face_bgr"] for d in recognizer.last_debug_faces
                if d.get("aligned_face_bgr") is not None
            ]
            if aligned_faces:
                emo_start = time.perf_counter_ns()
                emotion_classifier.classify_batch(aligned_faces)
                cuda_sync(device)
                emotion_ms = (time.perf_counter_ns() - emo_start) / 1_000_000.0
        cuda_sync(device)
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0
        now = time.perf_counter()
        # Per-stage timings populadas dentro de recognize_frame.
        stage = recognizer.last_timing or {}

        # Fora da janela cronometrada, entre t1 e a proxima leitura: casar deteccao
        # com anotacao custa microssegundos, mas nao entra na latencia por
        # construcao -- t1 ja foi lido acima.
        if avaliador is not None:
            avaliador.adicionar_quadro(indice_quadro, reconhecimentos)
        indice_quadro += 1

        if in_steady:
            rows.append(
                {
                    "video": video_path.name,
                    "latency_ms": latency_ms,
                    "detection_ms": float(stage.get("detection_ms", float("nan"))),
                    "alignment_ms": float(stage.get("alignment_ms", float("nan"))),
                    "embedding_ms": float(stage.get("embedding_ms", float("nan"))),
                    "search_ms": float(stage.get("search_ms", float("nan"))),
                    "emotion_ms": emotion_ms if emotion_classifier is not None else float("nan"),
                    "new_faces": float(stage.get("new_faces", float("nan"))),
                    "t_s": now - warmup_until,
                    "timestamp_ns": time.time_ns(),
                }
            )
            if loop_if_short and now >= steady_state_end:
                break

    release_video()
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"video": video_path.name, "n_frames": 0}

    duration_s = df["t_s"].iloc[-1] - df["t_s"].iloc[0]
    fps_mean = (len(df) - 1) / duration_s if duration_s > 0 else float("nan")
    lat = df["latency_ms"].to_numpy()
    summary = {
        "video": video_path.name,
        "n_frames": len(df),
        "fps_mean": float(fps_mean),
        "latency_median_ms": float(np.median(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_max_ms": float(np.max(lat)),
    }
    # Per-stage stats: median, P95, max (ms). Stages podem ter NaN se o frame
    # pulou a etapa (ex.: detection com temporal_cache hit) — ignoramos.
    for stage_col in ("detection_ms", "alignment_ms", "embedding_ms", "search_ms", "emotion_ms"):
        if stage_col not in df.columns:
            continue
        arr = df[stage_col].to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        base = stage_col.removesuffix("_ms")
        summary[f"{base}_median_ms"] = float(np.median(arr))
        summary[f"{base}_p95_ms"] = float(np.percentile(arr, 95))
        summary[f"{base}_max_ms"] = float(np.max(arr))
    for budget in LATENCY_BUDGETS_MS:
        frac = float((lat <= budget).mean())
        summary[f"frac_in_{int(budget)}ms"] = frac
    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hardware-id", required=True)
    p.add_argument("--videos", nargs="+", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--warmup-s", type=float, default=VIDEO_WARMUP_S)
    p.add_argument("--duration-s", type=float, default=MIN_VIDEO_DURATION_S)
    p.add_argument("--detector-backend", type=str, default="yunet", choices=["yunet", "mtcnn"])
    p.add_argument("--detector-conf", type=float, default=0.5)
    p.add_argument(
        "--yunet-input-long-side",
        type=int,
        default=0,
        help="Lado longo da entrada do YuNet. 0 (padrão) manda o quadro na "
             "resolução nativa, sem reamostrar — é o que o protocolo declara. "
             "Valores positivos reamostram para múltiplos de 32, o que nunca "
             "preserva os 800x600 do ChokePoint (a altura viraria 576).",
    )
    p.add_argument("--embedder-backend", type=str, default="arcface", choices=["arcface", "edgeface", "facelivt", "onnx", "tflite"])
    p.add_argument("--onnx-model", type=str, default=None,
                   help="modelo .onnx para --embedder-backend onnx")
    p.add_argument("--tflite-model", type=str, default=None,
                   help="modelo .tflite para --embedder-backend tflite")
    p.add_argument("--tflite-cpu", action="store_true",
                   help="desliga o delegate da Edge TPU no backend tflite")
    p.add_argument("--arcface-network", type=str, default="r100")
    p.add_argument("--embedder-precision", type=str, default="fp16")
    p.add_argument("--collection", type=str, default="Rostos")
    p.add_argument("--vector-store", type=str, default="numpy", choices=["numpy", "qdrant"],
                   help="backend da etapa de busca; use o mesmo em todos os alvos")
    p.add_argument("--gallery", type=str, default=None,
                   help="prefixo da galeria para --vector-store numpy (.npy + .json)")
    p.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    p.add_argument("--qdrant-api-key", type=str, default=None)
    p.add_argument("--score-threshold", type=float, default=0.45)
    p.add_argument("--max-faces", type=int, default=5)
    p.add_argument(
        "--groundtruth", type=Path, default=None,
        help="XML de anotação do ChokePoint (ex.: data/chokepoint/groundtruth/"
             "P1E_S1_C1.xml). Com ele, a rodada também calcula precision, recall "
             "e F1 de detecção e de reconhecimento em quality_metrics.json. "
             "Exige --no-loop e um único vídeo: o i-ésimo quadro tem de casar com "
             "a i-ésima entrada da anotação.",
    )
    p.add_argument(
        "--groundtruth-margem", type=float, default=None,
        help="Folga do casamento olho-dentro-da-caixa, em fração do tamanho da "
             "caixa (padrão 0.15). Só afeta as métricas, não a medição.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--no-loop",
        action="store_true",
        help="Não loopa o vídeo. Termina quando o último frame é processado, "
             "respeitando --duration-s apenas como teto.",
    )
    p.add_argument(
        "--allow-throttling",
        action="store_true",
        help="Grava a rodada mesmo se a placa acusar throttling na janela. Existe para "
             "alvos que nao conseguem sustentar a janela (a Pi 5 com fonte de 15 W acende "
             "sub-tensao sob carga nos 4 nucleos). A saida ganha RODADA_CONTAMINADA.txt "
             "com a fracao de amostras afetadas -- esses numeros nao entram na tabela "
             "comparativa sem essa ressalva.",
    )
    p.add_argument(
        "--with-emotion",
        action="store_true",
        help="Habilita classificação de emoção sobre cada face alinhada após "
             "recognize_frame; mede etapa separada em emotion_ms.",
    )
    p.add_argument(
        "--emotion-model",
        type=str,
        default=None,
        help="Path do modelo YOLO de emoção (.pt). Default: "
             "models/emotion/rafdb_yolo26n_best.pt.",
    )
    p.add_argument("--emotion-imgsz", type=int, default=224)
    p.add_argument(
        "--no-arcface-compile",
        action="store_true",
        help="Desliga torch.compile(reduce-overhead) no ArcFace. "
             "Necessário em Jetson Orin Nano (poucos SMs causam recompilação "
             "constante).",
    )
    p.add_argument(
        "--burst",
        action="store_true",
        help="Modo BURST: warm-up curto (3s, só pra passar a init do CUDA) e "
             "compile desligado, medindo o aparelho FRIO desde o início. Use com "
             "--duration-s curto (ex.: 20). Compare o fps_mean com uma rodada "
             "sustained (warm-up 60s + 300s) para quantificar o throttling.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from utils.arcface_inference import get_torch_device
    from utils.face_pipeline import FaceRecognizer

    device = args.device or get_torch_device()
    LOG.info("hardware=%s device=%s", args.hardware_id, device)

    # Modo burst: aparelho frio. Warm-up curto (só passa init do CUDA, sem deixar
    # aquecer) e compile OFF (senão a compilação JIT contamina a janela fria).
    warmup_s = args.warmup_s
    arcface_compile = not args.no_arcface_compile
    if args.burst:
        arcface_compile = False
        if args.warmup_s == VIDEO_WARMUP_S:   # usuário não sobrescreveu
            warmup_s = 3.0
        LOG.info("BURST: warmup=%.0fs, compile=off (medindo aparelho frio)", warmup_s)

    emotion_classifier = None
    if args.with_emotion:
        from utils.emotion_inference import DEFAULT_EMOTION_MODEL, EmotionClassifier
        emotion_classifier = EmotionClassifier(
            model_path=args.emotion_model or DEFAULT_EMOTION_MODEL,
            device=device,
            imgsz=args.emotion_imgsz,
            half=(device == "cuda"),
        )
        LOG.info("Emotion classifier loaded (device=%s, imgsz=%d)", device, args.emotion_imgsz)

    recognizer = FaceRecognizer(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_store=args.vector_store,
        gallery_path=args.gallery,
        onnx_model=args.onnx_model,
        tflite_model=args.tflite_model,
        tflite_use_tpu=not args.tflite_cpu,
        collection_name=args.collection,
        device=device,
        detector_backend=args.detector_backend,
        conf=args.detector_conf,
        yunet_input_long_side=args.yunet_input_long_side,
        embedder_backend=args.embedder_backend,
        arcface_network=args.arcface_network,
        embedder_precision=args.embedder_precision,
        score_threshold=args.score_threshold,
        max_faces=args.max_faces,
        temporal_cache=False,  # Benchmark wants raw per-frame latency.
        detect_interval=1,
        arcface_compile=arcface_compile,
        facelivt_compile=arcface_compile,
    )

    avaliador = None
    if args.groundtruth is not None:
        if not args.no_loop:
            raise SystemExit(
                "--groundtruth exige --no-loop: com loop o vídeo reinicia e o índice "
                "do quadro deixa de corresponder à anotação."
            )
        if len(args.videos) != 1:
            raise SystemExit("--groundtruth vale para um vídeo por vez.")
        gt = carregar_groundtruth(args.groundtruth)
        margem = args.groundtruth_margem if args.groundtruth_margem is not None else MARGEM_PADRAO
        avaliador = Avaliador(groundtruth=gt, margem=margem)
        LOG.info("groundtruth=%s | %d quadros, %d pessoas anotadas | margem=%.2f",
                 args.groundtruth, len(gt), sum(len(x) for x in gt), margem)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or result_dir(args.hardware_id, timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = SystemMonitor(args.hardware_id)
    reset_torch_peak_memory()
    monitor.start()
    steady_start = datetime.now(timezone.utc).isoformat()

    per_frame_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    try:
        for video in args.videos:
            df, summary = run_video(
                video_path=video,
                recognizer=recognizer,
                device=device,
                warmup_s=warmup_s,
                min_duration_s=args.duration_s,
                loop_if_short=not args.no_loop,
                emotion_classifier=emotion_classifier,
                avaliador=avaliador,
            )
            per_frame_frames.append(df)
            summary["hardware_id"] = args.hardware_id
            summaries.append(summary)
            LOG.info("[%s] %s", video.name, summary)
    finally:
        steady_end = datetime.now(timezone.utc).isoformat()
        mon_df = monitor.stop(check_throttling=not args.allow_throttling)

    if per_frame_frames:
        all_frames = pd.concat(per_frame_frames, ignore_index=True)
        all_frames["hardware_id"] = args.hardware_id
        all_frames.to_parquet(out_dir / "e2e_per_frame.parquet", index=False)

    pd.DataFrame(summaries).to_csv(out_dir / "e2e_video.csv", index=False)

    if avaliador is not None:
        metricas = avaliador.resumo()
        metricas["contexto"] = {
            "hardware_id": args.hardware_id,
            "detector": args.detector_backend,
            "extrator": args.embedder_backend,
            "modelo": args.onnx_model or args.tflite_model or args.arcface_network,
            "precisao": args.embedder_precision,
            "limiar_reconhecimento": args.score_threshold,
            "groundtruth": str(args.groundtruth),
            "video": str(args.videos[0]),
        }
        (out_dir / "quality_metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
        d = metricas["deteccao"]
        e2e = metricas["reconhecimento_e2e"]
        dado = metricas["reconhecimento_dado_deteccao"]
        LOG.info("deteccao      P=%.4f R=%.4f F1=%.4f  (tp=%d fp=%d fn=%d)",
                 d["precision"], d["recall"], d["f1"], d["tp"], d["fp"], d["fn"])
        LOG.info("reconh. e2e   P=%.4f R=%.4f F1=%.4f  macro-F1=%.4f  (%d/%d)",
                 e2e["micro_precision"], e2e["micro_recall"], e2e["micro_f1"],
                 e2e["macro_f1"], e2e["acertos"], e2e["casos"])
        LOG.info("reconh. |det  P=%.4f R=%.4f F1=%.4f  macro-F1=%.4f  (%d/%d)",
                 dado["micro_precision"], dado["micro_recall"], dado["micro_f1"],
                 dado["macro_f1"], dado["acertos"], dado["casos"])
        LOG.info("rank-1        dado_deteccao=%.4f  e2e=%.4f",
                 metricas["rank1"]["dado_deteccao"], metricas["rank1"]["e2e"])
    mon_df.to_parquet(out_dir / "system_metrics.parquet", index=False)

    info = build_system_info(
        args.hardware_id,
        detector_backend=args.detector_backend,
        embedder_backend=args.embedder_backend,
        qdrant_url=args.qdrant_url,
    )
    info.steady_state_start_utc = steady_start
    info.steady_state_end_utc = steady_end
    info.to_json(out_dir / "system_info.json")

    torch_mem = monitor.torch_memory_snapshot()
    (out_dir / "torch_memory.json").write_text(json.dumps(torch_mem, indent=2))

    # Carimbo da rodada salva sob --allow-throttling: sem ele o diretorio fica
    # indistinguivel de uma rodada limpa tres meses depois.
    if args.allow_throttling and "throttled_now" in mon_df.columns:
        agora = mon_df["throttled_now"].fillna(0)
        novo = mon_df.get("throttled_ever", pd.Series(dtype=float)).fillna(0)
        frac = float(agora.mean()) if len(agora) else float("nan")
        if frac > 0 or (len(novo) and novo.max() > 0):
            texto = (
                "Rodada gravada com --allow-throttling: a placa acusou throttling DENTRO "
                "da janela medida.\n"
                f"amostras de sistema      : {len(mon_df)}\n"
                f"fracao com throttled_now : {frac:.3f}\n"
                f"bit historico novo       : {'sim' if len(novo) and novo.max() > 0 else 'nao'}\n"
                f"temp max (C)             : {float(mon_df['cpu_temp_c'].max()):.1f}\n"
                "\nNao use estes numeros como medida limpa do sistema.\n"
            )
            (out_dir / "RODADA_CONTAMINADA.txt").write_text(texto)
            LOG.warning("throttling em %.1f%% das amostras -- %s",
                        frac * 100, out_dir / "RODADA_CONTAMINADA.txt")
    LOG.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
