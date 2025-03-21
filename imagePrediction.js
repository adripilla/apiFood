const tf = require("@tensorflow/tfjs");
const mobilenet = require("@tensorflow-models/mobilenet");
const fs = require("fs");
const { createCanvas, loadImage } = require("canvas");

// Variable global para el modelo MobileNet
let model;

// Cargar el modelo MobileNet
const loadModel = async () => {
  try {
    model = await mobilenet.load();
    console.log("✅ Modelo MobileNet cargado correctamente.");
  } catch (error) {
    console.error("❌ Error al cargar el modelo MobileNet:", error);
  }
};

// Función para analizar imagen
const analyzeImage = async (imagePath) => {
  try {
    if (!model) {
      console.error("❌ Error: El modelo MobileNet aún no está cargado.");
      return "Model not loaded";
    }

    // Cargar la imagen con canvas
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, image.width, image.height);

    // Convertir imagen a tensor
    const imageTensor = tf.browser.fromPixels(canvas);

    // Obtener predicciones del modelo
    const predictions = await model.classify(imageTensor);

    // 🔍 Imprimir predicciones en consola
    let food = null;

    predictions.forEach(pred => {
        if ((pred.probability * 100).toFixed(2) > 80) {
            food = pred.className;
        }
    });

    // Filtrar resultados y devolver el de mayor probabilidad
    if(food)
        return food;
    else    
        return "No match found";
  } catch (error) {
    console.error("❌ Error al procesar la imagen:", error);
    return null;
  }
};

module.exports = { analyzeImage, loadModel };
