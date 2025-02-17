from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import io
import os
import requests
import re
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
options.add_argument("--headless")  # Ejecutar en segundo plano

driver = webdriver.Edge(options=options)

# Cargar alimentos desde un archivo de texto
def cargar_alimentos(archivo="alimentos.txt"):
    if not os.path.exists(archivo):
        with open(archivo, "w", encoding="utf-8") as f:
            f.write("apple\nbanana\ncarrot\nbroccoli\ncucumber\ntomato\nonion\ngarlic\n")
    with open(archivo, "r", encoding="utf-8") as f:
        return [f"a {line.strip()}" for line in f if line.strip()]

# Leer alimentos desde el archivo
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

def get_nutrition_info(predicted_class):
    driver.get("https://developer.edamam.com/edamam-nutrition-api-demo")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "demoAnalysis")))
    textarea = driver.find_element(By.ID, "demoAnalysis")
    textarea.clear()
    textarea.send_keys(predicted_class)
    
    analyze_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "calc-analysis-api")))
    driver.execute_script("arguments[0].scrollIntoView();", analyze_button)
    ActionChains(driver).move_to_element(analyze_button).click().perform()
    
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "table")))
    table = driver.find_element(By.CLASS_NAME, "table")
    rows = table.find_elements(By.TAG_NAME, "tr")[1:]
    
    nutrition_data = [{
        "Qty": row.find_elements(By.TAG_NAME, "th")[0].text,
        "Unit": row.find_elements(By.TAG_NAME, "th")[1].text,
        "Food": row.find_elements(By.TAG_NAME, "th")[2].text,
        "Calories": row.find_elements(By.TAG_NAME, "th")[3].text,
        "Weight": row.find_elements(By.TAG_NAME, "th")[4].text
    } for row in rows if len(row.find_elements(By.TAG_NAME, "th")) >= 5]
    
    return nutrition_data

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    image = Image.open(io.BytesIO(file.read()))
    predicted_class = analyze_image(image)
    print(predicted_class)
    
    search_query = re.sub(r'^(a|an)\s+', '', predicted_class, flags=re.IGNORECASE)
    nutrition_info = get_nutrition_info(predicted_class)
    
    print(nutrition_info)
     
    regex = re.compile(re.escape(search_query), re.IGNORECASE)
    
    # Obtener datos de la API mealdb_search_url con manejo de errores
    mealdb_search_url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={search_query}"
    try:
        search_response = requests.get(mealdb_search_url)
        mealdb_search_data = search_response.json()
        
        if "meals" in mealdb_search_data:
            filtered_search = [meal for meal in mealdb_search_data["meals"] if regex.search(meal.get("strMeal", ""))]
        else:
            filtered_search = "none"
    except Exception as e:
        mealdb_search_data = {"error": str(e)}
        filtered_search = "none"

   

    return jsonify({
        "predicted_class": predicted_class,
        "nutrition_info": nutrition_info,
        "mealdb_search": filtered_search
    }), 200



if __name__ == "__main__":
    app.run(debug=True)
