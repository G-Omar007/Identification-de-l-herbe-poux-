from flask import Flask, request, jsonify
from flask_cors import CORS
from classifier import predict
from inaturalist import get_species_info, get_nearby_observations

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.75
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mo
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _extension_valide(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "Image manquante"}), 400

    image_file = request.files["file"]

    if image_file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400

    if not _extension_valide(image_file.filename):
        return jsonify({"error": "Format non supporté. Utilisez JPG, PNG ou WebP."}), 400

    image_bytes = image_file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "Fichier trop volumineux (max 10 Mo)"}), 413

    lat = request.form.get("lat", type=float)
    lng = request.form.get("lng", type=float)

    try:
        resultat = predict(image_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if resultat["is_ragweed"]:
        if resultat["confidence"] >= 0.85:
            resultat["alerte"] = "ELEVE"
        elif resultat["confidence"] >= CONFIDENCE_THRESHOLD:
            resultat["alerte"] = "MOYEN"
        else:
            resultat["alerte"] = "INCERTAIN"
    else:
        resultat["alerte"] = "AUCUN"

    if resultat["is_ragweed"] and resultat["confidence"] >= CONFIDENCE_THRESHOLD:
        info = get_species_info()
        if info:
            resultat["info_espece"] = info
        if lat is not None and lng is not None:
            resultat["observations_proches"] = get_nearby_observations(lat, lng)

    return jsonify(resultat), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modele": "YOLOv11n-cls"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
