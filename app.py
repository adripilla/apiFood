from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import io
import os
import requests
from flask_cors import CORS
from requests_oauthlib import OAuth1

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

# ----- Configuración para FatSecret con OAuth 1.0 -----
FATSECRET_CLIENT_ID = "bf05bb842a994833b309068c4f010103"  # Reemplaza con tu Consumer Key
FATSECRET_CLIENT_SECRET = "99b020f56caf4e2aabfc7437bbe5ed8f"  # Reemplaza con tu Consumer Secret

def get_nutrition_info_fatsecret(food_name):
    url = "https://platform.fatsecret.com/rest/server.api"
    params = {
        "method": "food.search",
        "search_expression": food_name,
        "format": "json"
    }
    # Se utiliza OAuth1 para firmar la solicitud
    auth = OAuth1(FATSECRET_CLIENT_ID,
                  client_secret=FATSECRET_CLIENT_SECRET,
                  signature_method='HMAC-SHA1',
                  signature_type='auth_header')
    response = requests.get(url, params=params, auth=auth)
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            return {"error": "Error al decodificar JSON", "details": str(e)}
    else:
        return {"error": "Error al obtener información nutricional", "details": response.text}

# ----- Endpoint para analizar imagen usando CLIP -----
@app.route("/analyze_image", methods=["POST"])
def analyze_image_endpoint():
    print("analyze")
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    image = Image.open(io.BytesIO(file.read()))
    predicted_class = analyze_image(image)
    print("funciona")
    return jsonify({"predicted_class": predicted_class}), 200

# ----- Nuevo endpoint que integra la API de FatSecret -----
@app.route("/predict_nutrition", methods=["POST"])
def predict_nutrition():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Analiza la imagen para determinar el alimento
    image = Image.open(io.BytesIO(file.read()))
    predicted_class = analyze_image(image)
    
    # Usa la API de FatSecret (OAuth1) para obtener información nutricional
    nutrition_info = get_nutrition_info_fatsecret(predicted_class)
    
    return jsonify({
        "predicted_class": predicted_class,
        "nutrition_info": nutrition_info
    }), 200

if __name__ == "__main__":
    # Configuración para producción en la nube
    app.run(debug=False, host="0.0.0.0", port=5000)
