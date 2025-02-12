from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Función para preprocesar y analizar la imagen
def analyze_image(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    texts = ["a photo of a dog", "a photo of a cat", "a picture of a car"]
    text_inputs = torch.cat([clip.tokenize(text) for text in texts]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)

    best_match = torch.argmax(similarity).item()
    return texts[best_match]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Leer la imagen enviada
    image = Image.open(io.BytesIO(file.read()))
    
    # Procesar y analizar la imagen
    prediction = analyze_image(image)

    return jsonify({"predicted_class": prediction}), 200

if __name__ == "__main__":
    app.run(debug=True)
