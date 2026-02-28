from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from web.backend.inference import load_model, predict

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=None)
model = load_model()


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "image file missing"}), 400

    file = request.files["image"]
    prediction = predict(model, file)
    return jsonify({"result": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
