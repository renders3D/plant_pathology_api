from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os

app = FastAPI(
    title="Plant Pathology Inference API",
    description="API para clasificación de patologías en cultivos vía Drone/App",
    version="1.0.0"
)

# --- CONFIGURACIÓN ACTUALIZADA ---
MODEL_FILENAME = "plant_model.keras"
MODEL_PATH = f"models/{MODEL_FILENAME}"
class_names = ['deficiencia', 'fusario', 'sanas'] 
img_shape = 256

# Variable global para el modelo
model = None

@app.on_event("startup")
def load_model():
    """Carga el modelo en memoria al iniciar el contenedor"""
    global model
    print(f"🔄 Intentando cargar modelo desde: {MODEL_PATH}")
    
    # Depuración de archivos
    if os.path.exists("models"):
        print(f"📂 Archivos detectados en 'models': {os.listdir('models')}")
    else:
        print("⚠️ ALERTA: La carpeta 'models' no existe en el contenedor.")

    try:
        # Carga robusta para .keras
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ ¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"❌ Error fatal cargando el modelo: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((img_shape, img_shape))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando la imagen.")

@app.get("/")
def home():
    return {"status": "online", "docs_url": "http://localhost:8000/docs"}

# --- AQUÍ ESTABA EL DETALLE CLAVE ---
# Debemos usar File(...) explícitamente para que Swagger muestre el botón de subir
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está listo.")
    
    # Leer y procesar
    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    # Inferencia
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    result_class = class_names[class_idx]

    return {
        "prediction": result_class,
        "confidence": round(confidence * 100, 2),
        "filename": file.filename
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
