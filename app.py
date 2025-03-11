from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

def cargar_alimentos(archivo="alimentos.txt"):
    if not os.path.exists(archivo):
        with open(archivo, "w", encoding="utf-8") as f:
            f.write("apple\nbanana\ncarrot\nbroccoli\ncucumber\ntomato\nonion\ngarlic\n")
    with open(archivo, "r", encoding="utf-8") as f:
        return [f"a {line.strip()}" for line in f if line.strip()]

# Cargar la lista de alimentos
alimentos = cargar_alimentos()

def analyze_image(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(text) for text in alimentos]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)

    best_match = torch.argmax(similarity).item()
    return alimentos[best_match]

@app.route("/analyze_image", methods=["POST"])
def analyze_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    image = Image.open(io.BytesIO(file.read()))
    predicted_class = analyze_image(image)
    return jsonify({"predicted_class": predicted_class}), 200

if __name__ == "__main__":
    # Configuración para producción en la nube
    app.run(debug=False, host="0.0.0.0", port=80)
