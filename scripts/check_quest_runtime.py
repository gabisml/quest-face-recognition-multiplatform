import sys

print("=" * 60)
print(" Quest Facial Recognition - Runtime Check")
print("=" * 60)

print("\n[Python]")
print(sys.version)

print("\n[PyTorch]")
try:
    import torch
    print("Version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("ERRO:", exc)

print("\n[ONNX Runtime]")
try:
    import onnxruntime as ort
    print("Version:", ort.__version__)
    print("Providers:", ort.get_available_providers())

    providers = ort.get_available_providers()

    if "CUDAExecutionProvider" not in providers and "TensorrtExecutionProvider" not in providers:
        print("AVISO: ONNX Runtime nao possui provider GPU.")
except Exception as exc:
    print("ERRO:", exc)

print("\n[OpenCV]")
try:
    import cv2
    print("Version:", cv2.__version__)
    print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
except Exception as exc:
    print("ERRO:", exc)

print("\n[Qdrant]")
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection("Rostos")

    print("Collection: Rostos")
    print("Points:", info.points_count)
except Exception as exc:
    print("AVISO:", exc)

print("\n[Models]")

from pathlib import Path

models = [
    Path("models/yunet/face_detection_yunet_2023mar.onnx"),
    Path("models/onnx/arcface_r100.onnx"),
]

for model in models:
    if model.exists():
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"OK: {model} ({size_mb:.1f} MB)")
    else:
        print(f"FALTANDO: {model}")

print("\n" + "=" * 60)
print(" Runtime check concluido")
print("=" * 60)
