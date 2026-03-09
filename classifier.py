# classifier.py
# Inférence YOLOv11n-cls via ONNX — même logique que votre Colab
# Classes confirmées : 0=ambrosia, 1=non_ambrosia

import io
import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Classes (ordre confirmé par les métadonnées du modèle) ──
CLASS_NAMES = ["ambrosia", "non_ambrosia"]
AMBROSIA_IDX = 0

# ── Chargement unique au démarrage du serveur ──
print("[Classifier] Chargement du modèle ONNX...")
_session = ort.InferenceSession(
    "meilleur.onnx",
    providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name
print(f"[Classifier] Modèle prêt. Input: {_input_name} | Classes: {CLASS_NAMES}")


def predict(image_bytes: bytes) -> dict:
    """
    Reçoit une image en bytes (JPEG/PNG), retourne la prédiction.
    Même logique de prétraitement que votre notebook Colab.
    """
    # 1. Ouvrir et redimensionner (identique à Colab : 224x224 RGB)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    # 2. Normalisation [0,255] → [0.0, 1.0]
    arr = np.array(img, dtype=np.float32) / 255.0

    # 3. [H,W,C] → [1,C,H,W]  (format attendu par YOLO ONNX)
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    # 4. Inférence
    outputs = _session.run(None, {_input_name: arr})
    logits  = outputs[0][0]

    # 5. Softmax pour obtenir des probabilités
    exp    = np.exp(logits - logits.max())   # stabilité numérique
    probs  = exp / exp.sum()

    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence":      round(top_conf, 4),
        "is_ragweed":      top_idx == AMBROSIA_IDX,
        "all_scores": {
            "ambrosia":     round(float(probs[AMBROSIA_IDX]), 4),
            "non_ambrosia": round(float(probs[1]), 4),
        }
    }
