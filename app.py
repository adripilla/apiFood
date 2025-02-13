from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import io
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Configurar Selenium con Edge
options = webdriver.EdgeOptions()
options.add_argument("--headless")  # Ejecutar en segundo plano (opcional)
driver = webdriver.Edge(options=options)

# Función para preprocesar y analizar la imagen
def analyze_image(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    texts = [
        "an apple", "an avocado", "a banana", "a carrot", "a broccoli", "a cucumber", "a tomato", "an onion", "a garlic",
        "a chickpea", "another food","a pizza"
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

# Obtener información nutricional desde Edamam
def get_nutrition_info(predicted_class):
    driver.get("https://developer.edamam.com/edamam-nutrition-api-demo")

    # Esperar a que la página cargue completamente
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "demoAnalysis"))
    )

    # Encontrar el área de texto y escribir el alimento detectado
    textarea = driver.find_element(By.ID, "demoAnalysis")
    textarea.clear()
    textarea.send_keys(predicted_class)

    # Encontrar el botón y hacer clic con precaución
    analyze_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "calc-analysis-api"))
    )

    # Asegurar que el botón sea visible y hacer clic
    driver.execute_script("arguments[0].scrollIntoView();", analyze_button)
    ActionChains(driver).move_to_element(analyze_button).click().perform()

    # Esperar los resultados de la tabla
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "table"))
    )

    # Extraer los datos de la tabla
    table = driver.find_element(By.CLASS_NAME, "table")
    rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Omitir la cabecera

    nutrition_data = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "th")
        if len(cols) >= 5:
            nutrition_data.append({
                "Qty": cols[0].text,
                "Unit": cols[1].text,
                "Food": cols[2].text,
                "Calories": cols[3].text,
                "Weight": cols[4].text,
            })

    return nutrition_data

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
    predicted_class = analyze_image(image)
    
    # Obtener información nutricional desde Edamam
    nutrition_info = get_nutrition_info(predicted_class)

    # Regresar el resultado y la información
    return jsonify({
        "predicted_class": predicted_class,
        "nutrition_info": nutrition_info
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
