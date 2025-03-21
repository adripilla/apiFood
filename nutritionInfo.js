const crypto = require("crypto");
const OAuth = require("oauth-1.0a");
const axios = require("axios");

require("dotenv").config();

const FATSECRET_CLIENT_ID = process.env.FATSECRET_CLIENT_ID;
const FATSECRET_CLIENT_SECRET = process.env.FATSECRET_CLIENT_SECRET;

// Configurar OAuth 1.0a
const oauth = OAuth({
  consumer: {
    key: FATSECRET_CLIENT_ID,
    secret: FATSECRET_CLIENT_SECRET,
  },
  signature_method: "HMAC-SHA1",
  hash_function(base_string, key) {
    return crypto.createHmac("sha1", key).update(base_string).digest("base64");
  },
});

// Función para obtener detalles por food_id
const getFoodDetailsById = async (foodId) => {
  const url = "https://platform.fatsecret.com/rest/server.api";

  const params = {
    method: "food.get.v4",
    food_id: foodId,
    format: "json",
  };

  const requestData = {
    url,
    method: "GET",
    data: params,
  };

  const oauthParams = oauth.authorize(requestData);
  const allParams = { ...params, ...oauthParams };

  const response = await axios.get(url, {
    params: allParams,
  });

  return response.data;
};

// 🔍 Función principal para obtener información nutricional
const getNutritionInfoFatSecret = async (foodName) => {
  const url = "https://platform.fatsecret.com/rest/server.api";

  if (foodName == "No match found") {
    return { error: "No food name provided"};
    }
  const params = {
    method: "foods.search",
    search_expression: foodName,
    format: "json",
  };

  const requestData = {
    url,
    method: "GET",
    data: params,
  };

  const oauthParams = oauth.authorize(requestData);
  const allParams = { ...params, ...oauthParams };

  try {
    const searchResponse = await axios.get(url, {
      params: allParams,
    });

    const foodIds = searchResponse.data.foods.food.map(item => item.food_id);
    console.log("🔎 Primer food_id:", foodIds[0]);

    // Obtener detalles del primer resultado
    const nutritionDetails = await getFoodDetailsById(foodIds[0]);
    return nutritionDetails;

  } catch (error) {
    console.error("❌ Error al obtener información nutricional:", error.response?.data || error.message);
    return { error: "Error fetching nutrition data" };
  }
};

module.exports = { getNutritionInfoFatSecret };