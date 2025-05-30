const tf = require('@tensorflow/tfjs-node');

const MODEL_URL = 'https://shimmering-babka-d65c69.netlify.app/model.json';
let model = null;

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel(MODEL_URL);
    console.log("✅ Modelo cargado desde URL externa");
  }
}

exports.handler = async (event) => {
  try {
    await loadModel();

    const body = JSON.parse(event.body);
    const base64 = body.image;

    if (!base64) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Falta la imagen" })
      };
    }

    const buffer = Buffer.from(base64, 'base64');
    const imageTensor = tf.node.decodeImage(buffer)
      .resizeNearestNeighbor([300, 300])
      .toFloat()
      .div(tf.scalar(255))
      .expandDims();

    const prediction = await model.predict(imageTensor).data();

    return {
      statusCode: 200,
      body: JSON.stringify({
        prediction: Array.from(prediction),
        labels: ["Mascarilla Correcta", "Mascarilla Incorrecta", "Sin Mascarilla"]
      })
    };
  } catch (error) {
    console.error("❌ Error en predicción:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
};

