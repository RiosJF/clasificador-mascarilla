const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const app = express();
app.use(express.json({ limit: '10mb' }));

let model;

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel('https://shimmering-babka-d65c69.netlify.app/model.json');
    console.log("✅ Modelo cargado");
  }
}

app.post('/predict', async (req, res) => {
  try {
    await loadModel();

    const base64 = req.body.image;
    if (!base64) return res.status(400).json({ error: "Falta la imagen" });

    const buffer = Buffer.from(base64, 'base64');
    const imageTensor = tf.node.decodeImage(buffer)
      .resizeNearestNeighbor([300, 300])
      .toFloat()
      .div(tf.scalar(255))
      .expandDims();

    const prediction = await model.predict(imageTensor).data();
    res.json({
      prediction: Array.from(prediction),
      labels: ["Mascarilla Correcta", "Mascarilla Incorrecta", "Sin Mascarilla"]
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = app;
