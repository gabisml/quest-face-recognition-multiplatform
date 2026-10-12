"""Consolida os quality_metrics.json de várias rodadas numa tabela.

Uso:
    python -m bench.aggregate_quality results/rtx3090/*/ --out results/_aggregated/qualidade.csv

Cada rodada contribui uma linha, identificada por hardware, detector, modelo e
precisão -- os campos que o e2e_video gravou em `contexto`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

COLUNAS = [
    "hardware", "detector", "modelo", "precisao",
    "pessoas_anotadas", "det_tp", "det_fp", "det_fn",
    "det_precision", "det_recall", "det_f1",
    "rec_e2e_micro_f1", "rec_e2e_macro_f1", "rec_e2e_acertos", "rec_e2e_casos",
    "rec_det_micro_f1", "rec_det_macro_f1", "rec_det_acertos", "rec_det_casos",
    "rank1_dado_deteccao", "rank1_e2e", "desconhecido",
    "identidades", "perdidas_iod_mediana", "diretorio",
]


def _nome_modelo(bruto: Optional[str]) -> str:
    if not bruto:
        return ""
    return Path(bruto).stem


def linha(caminho: Path) -> dict[str, Any]:
    m = json.loads(caminho.read_text(encoding="utf-8"))
    ctx = m.get("contexto", {})
    d = m["deteccao"]
    e2e = m["reconhecimento_e2e"]
    dado = m["reconhecimento_dado_deteccao"]
    r1 = m["rank1"]
    return {
        "hardware": ctx.get("hardware_id", ""),
        "detector": ctx.get("detector", ""),
        "modelo": _nome_modelo(ctx.get("modelo")),
        "precisao": ctx.get("precisao", ""),
        "pessoas_anotadas": m["pessoas_anotadas"],
        "det_tp": d["tp"], "det_fp": d["fp"], "det_fn": d["fn"],
        "det_precision": d["precision"], "det_recall": d["recall"], "det_f1": d["f1"],
        "rec_e2e_micro_f1": e2e["micro_f1"], "rec_e2e_macro_f1": e2e["macro_f1"],
        "rec_e2e_acertos": e2e["acertos"], "rec_e2e_casos": e2e["casos"],
        "rec_det_micro_f1": dado["micro_f1"], "rec_det_macro_f1": dado["macro_f1"],
        "rec_det_acertos": dado["acertos"], "rec_det_casos": dado["casos"],
        "rank1_dado_deteccao": r1["dado_deteccao"], "rank1_e2e": r1["e2e"],
        "desconhecido": m["previstas_desconhecido"],
        "identidades": e2e["identidades_na_referencia"],
        "perdidas_iod_mediana": m["faces_perdidas_iod"]["mediana"],
        "diretorio": caminho.parent.name,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path,
                   help="diretórios de rodada, ou os próprios quality_metrics.json")
    p.add_argument("--out", type=Path, default=None, help="CSV de saída")
    args = p.parse_args()

    arquivos = []
    for d in args.dirs:
        if d.is_file():
            arquivos.append(d)
        elif (d / "quality_metrics.json").exists():
            arquivos.append(d / "quality_metrics.json")
    if not arquivos:
        raise SystemExit("nenhum quality_metrics.json encontrado nos caminhos dados")

    linhas = sorted(
        (linha(a) for a in arquivos),
        key=lambda r: (r["hardware"], r["detector"], r["modelo"], r["precisao"]),
    )

    largura = {c: max(len(c), *(len(str(r[c])) for r in linhas)) for c in COLUNAS if c != "diretorio"}
    mostrar = ["detector", "modelo", "precisao", "det_precision", "det_recall", "det_f1",
               "rec_e2e_micro_f1", "rec_det_micro_f1", "rank1_dado_deteccao", "desconhecido"]
    print("  ".join(c.ljust(largura[c]) for c in mostrar))
    for r in linhas:
        print("  ".join(str(r[c]).ljust(largura[c]) for c in mostrar))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUNAS)
            w.writeheader()
            w.writerows(linhas)
        print(f"\n{len(linhas)} linhas -> {args.out}")


if __name__ == "__main__":
    main()
