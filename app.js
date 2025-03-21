const express = require("express");
const multer = require("multer");
const dotenv = require("dotenv");
const { analyzeImage, loadModel } = require("./imagePrediction");
const { getNutritionInfoFatSecret } = require("./nutritionInfo");
const fs = require("fs");

// Cargar variables de entorno
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Configuración de almacenamiento de imágenes
const upload = multer({ dest: "uploads/" });

// 🚀 Cargar el modelo MobileNet antes de iniciar el servidor
loadModel()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`🚀 Servidor corriendo en http://localhost:${PORT}`);
    });
  })
  .catch((error) => {
    console.error("❌ Error al cargar el modelo MobileNet:", error);
  });

// ----- Endpoint para analizar imágenes -----
app.post("/analyze_image", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const imagePath = req.file.path;
  const predictedClass = await analyzeImage(imagePath);

  fs.unlinkSync(imagePath); // Eliminar archivo temporal

  if (!predictedClass) {
    return res.status(500).json({ error: "Error analyzing image" });
  }

  res.json({ predicted_class: predictedClass });
});

// ----- Endpoint para predecir nutrición -----
app.post("/predict_nutrition", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const imagePath = req.file.path;
  const predictedClass = await analyzeImage(imagePath);

  fs.unlinkSync(imagePath); // Eliminar archivo temporal

  if (!predictedClass) {
    return res.status(500).json({ error: "Error analyzing image" });
  }

  const nutritionInfo = await getNutritionInfoFatSecret(predictedClass);

  res.json({ predicted_class: predictedClass, nutrition_info: nutritionInfo });
});


