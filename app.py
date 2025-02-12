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
    texts = [
    "a apple", "a avocado", "a banana", "a carrot", "a broccoli", "a cucumber", "a tomato", "an onion", "a garlic", 
    "a potato", "a spinach", "a kale", "a lettuce", "a cabbage", "an eggplant", "a zucchini", "a bell pepper", 
    "a mushroom", "an asparagus", "a celery", "a cauliflower", "a sweet potato", "a green bean", "a pea", "a corn", 
    "an artichoke", "a pumpkin", "a squash", "a beet", "a chard", "a radish", "a leek", "a parsnip", "a fennel", 
    "a brussels sprout", "an okra", "a rhubarb", "a green onion", "a chive", "a basil", "a cilantro", "a parsley", 
    "a mint", "a dill", "a oregano", "a thyme", "a sage", "a rosemary", "a tarragon", "a bay leaf", "a marjoram", 
    "a lavender", "a paprika", "a cumin", "a turmeric", "a cinnamon", "a nutmeg", "a clove", "a ginger", "a cardamom", 
    "a allspice", "a chili powder", "a cayenne pepper", "a black pepper", "a salt", "a soy sauce", "a honey", 
    "a maple syrup", "a brown sugar", "a white sugar", "a molasses", "a agave nectar", "a coconut sugar", "a stevia", 
    "a almond", "a walnut", "a cashew", "a peanut", "a hazelnut", "a pistachio", "a sunflower seed", "a pumpkin seed", 
    "a chia seed", "a flax seed", "a sesame seed", "a quinoa", "a rice", "an oat", "a barley", "a wheat", "a spelt", 
    "a rye", "a cornmeal", "a polenta", "a buckwheat", "a amaranth", "a millet", "a tofu", "a tempeh", "a chickpea","a other food"
]

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
