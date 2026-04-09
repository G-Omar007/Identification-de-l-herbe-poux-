from flask import Flask, request, jsonify
from classifier import predict
from inaturalist import get_species_info, get_nearby_observations

app = Flask(__name__)

CONFIDENCE_THRESHOLD = 0.75


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Identificateur d'herbe à poux",
        "version": "1.0",
        "status": "en ligne",
        "routes": {
            "POST /classify": "Classifier une image (champ: file, lat, lng)",
            "GET /health": "Statut du service"
        }
    }), 200


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "Image manquante"}), 400

    image_file = request.files["file"]

    if image_file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400

    lat = request.form.get("lat", type=float)
    lng = request.form.get("lng", type=float)

    try:
        image_bytes = image_file.read()
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
        if lat and lng:
            resultat["observations_proches"] = get_nearby_observations(lat, lng)

    return jsonify(resultat), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modele": "YOLOv11n-cls"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
