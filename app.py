from flask import Flask, request, jsonify
from classifier import predict
from inaturalist import get_species_info, get_nearby_observations

app = Flask(__name__)

CONFIDENCE_THRESHOLD = 0.75

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
            resultat["alerte"] = "ÉLEVÉ"
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

@app.route("/health",
