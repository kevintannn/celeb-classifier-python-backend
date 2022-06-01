from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)
# cors = CORS(app, {r"*/*": {"origins": "*"}})


@app.route("/")
def none():
    return "Welcome to Celebrity Image Classifier API"


@app.route("/classify_image", methods=["GET", "POST"])
def classify_image():
    image_data = request.form["image_data"]

    response = jsonify(util.classify_image(image_data))

    return response


if __name__ == "__main__":
    print("Starting python flask server...")
    util.load_saved_artifacts()
    app.run()
