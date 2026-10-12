"""YuNet 120x160 em TFLite, com decode dos priors, rodando no CPU ou na Edge TPU.

O modelo vem de PINTO_model_zoo/144_YuNet (conversao do YuNet 2021 do opencv_zoo
via openvino2tensorflow). Saidas cruas, 1076 priors:
    loc  [1076, 14]  -> 4 de bbox + 10 de landmark (5 pontos)
    conf [1076,  2]  -> softmax fundo/face
    iou  [1076,  1]  -> score de IoU
O decode (priors + NMS) reproduz o de Kazuhito00/YuNet-ONNX-TFLite-Sample, que e
a referencia citada pelo proprio PINTO, porque o cv2.FaceDetectorYN faz isso em
C++ e nao aceita TFLite.

Uso:
    python yunet_tflite_tpu.py MODELO.tflite IMAGEM.jpg [--edgetpu] [--runs N]
"""
import argparse
import time
from itertools import product

import cv2
import numpy as np

MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STEPS = [8, 16, 32, 64]
VARIANCE = [0.1, 0.2]


def make_interpreter(model_path, use_edgetpu):
    """tflite_runtime (placa) ou tensorflow.lite (WSL), com ou sem delegate."""
    try:
        import tflite_runtime.interpreter as tfl
    except ImportError:
        import tensorflow.lite as tfl
    if use_edgetpu:
        return tfl.Interpreter(model_path=model_path,
                               experimental_delegates=[tfl.load_delegate("libedgetpu.so.1")])
    return tfl.Interpreter(model_path=model_path)


def generate_priors(w, h):
    """Feature maps por halving sucessivo: [15,20] [7,10] [3,5] [1,2] -> 1076 priors."""
    fm2 = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
    fms = []
    cur = fm2
    for _ in range(4):
        cur = [int(cur[0] / 2), int(cur[1] / 2)]
        fms.append(cur)
    priors = []
    for k, f in enumerate(fms):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in MIN_SIZES[k]:
                priors.append([(j + 0.5) * STEPS[k] / w, (i + 0.5) * STEPS[k] / h,
                               min_size / w, min_size / h])
    return np.array(priors, dtype=np.float32)


def decode(loc, conf, iou, priors, w, h):
    cls_scores = conf[:, 1]
    iou_scores = np.clip(iou[:, 0], 0.0, 1.0)
    scores = np.sqrt(cls_scores * iou_scores)[:, None]
    scale = np.array([w, h], dtype=np.float32)
    bboxes = np.hstack((
        (priors[:, 0:2] + loc[:, 0:2] * VARIANCE[0] * priors[:, 2:4]) * scale,
        (priors[:, 2:4] * np.exp(loc[:, 2:4] * VARIANCE)) * scale,
    ))
    bboxes[:, 0:2] -= bboxes[:, 2:4] / 2
    lms = np.hstack([
        (priors[:, 0:2] + loc[:, 4 + 2 * i:6 + 2 * i] * VARIANCE[0] * priors[:, 2:4]) * scale
        for i in range(5)
    ])
    return np.hstack((bboxes, lms, scores))


class YuNet:
    def __init__(self, model_path, use_edgetpu=False, conf_th=0.6, nms_th=0.3):
        self.it = make_interpreter(model_path, use_edgetpu)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.h, self.w = int(self.inp["shape"][1]), int(self.inp["shape"][2])
        self.priors = generate_priors(self.w, self.h)
        self.conf_th, self.nms_th = conf_th, nms_th
        # identifica as saidas pelo shape, nao pela ordem (o compilador permuta)
        self.out_by_last = {}
        for o in self.it.get_output_details():
            self.out_by_last[int(o["shape"][-1])] = o
        assert set(self.out_by_last) == {14, 2, 1}, f"saidas inesperadas: {[list(o['shape']) for o in self.it.get_output_details()]}"

    def _get(self, last_dim):
        o = self.out_by_last[last_dim]
        t = self.it.get_tensor(o["index"]).astype(np.float32)
        scale, zp = o["quantization"]
        if scale:
            t = (t - zp) * scale
        return t

    def set_input(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = small.reshape(1, self.h, self.w, 3)
        if self.inp["dtype"] == np.float32:
            x = x.astype(np.float32)
        else:
            x = x.astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)

    def invoke(self):
        self.it.invoke()

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        self.set_input(bgr)
        self.invoke()
        dets = decode(self._get(14), self._get(2), self._get(1), self.priors, self.w, self.h)
        keep = cv2.dnn.NMSBoxes(dets[:, 0:4].tolist(), dets[:, -1].tolist(),
                                self.conf_th, self.nms_th, top_k=5000)
        out = []
        if len(keep) > 0:
            sel = dets[np.array(keep).flatten()]
            sx, sy = W / self.w, H / self.h
            for d in sel:
                out.append({
                    "score": float(d[-1]),
                    "bbox": [int(d[0] * sx), int(d[1] * sy), int(d[2] * sx), int(d[3] * sy)],
                    "kps": (d[4:14].reshape(5, 2) * np.array([sx, sy])).astype(int),
                })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--edgetpu", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.6)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    y = YuNet(args.model, args.edgetpu, conf_th=args.conf)
    print(f"modelo : {args.model.split('/')[-1]}")
    print(f"input  : {y.w}x{y.h} {y.inp['dtype'].__name__} quant={np.round(y.inp['quantization'],6)}")
    print(f"priors : {len(y.priors)}")
    print(f"imagem : {img.shape[1]}x{img.shape[0]}")

    faces = y.detect(img)
    print(f"\nfaces (conf>={args.conf}): {len(faces)}")
    for f in faces:
        x, yy, w_, h_ = f["bbox"]
        print(f"  score={f['score']:.3f} bbox=({x},{yy})-({x+w_},{yy+h_}) kps={f['kps'].tolist()}")

    y.set_input(img)
    for _ in range(10):
        y.invoke()
    t = time.perf_counter()
    for _ in range(args.runs):
        y.invoke()
    dt = (time.perf_counter() - t) / args.runs
    print(f"\ninvoke  : {dt*1000:.2f} ms ({1/dt:.1f} FPS)")

    t = time.perf_counter()
    for _ in range(max(10, args.runs // 5)):
        y.detect(img)
    n = max(10, args.runs // 5)
    dte = (time.perf_counter() - t) / n
    print(f"detect(): {dte*1000:.2f} ms ({1/dte:.1f} FPS)  [resize + invoke + decode + NMS]")


if __name__ == "__main__":
    main()
