from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import requests  # Para descargar el modelo

app = FastAPI(
    title="Plant Pathology Inference API",
    description="API para clasificación de patologías en cultivos vía Drone/App",
    version="1.0.0"
)

# --- CONFIGURACIÓN ---
MODEL_FILENAME = "plant_model.keras"
LOCAL_MODEL_PATH = f"models/{MODEL_FILENAME}"

# URL de Azure Blob Storage
MODEL_URL = "https://plantmodelsstorage.blob.core.windows.net/models/plant_model.keras" 

class_names = ['deficiencia', 'fusario', 'sanas'] 
model = None

def download_model():
    """Descarga el modelo desde Azure si no existe localmente"""
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"⬇️ Modelo no encontrado localmente. Descargando desde Azure: {MODEL_URL}")
        print("Esto puede tardar unos minutos dependiendo de la conexión...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Lanza error si la URL está mal (404, 403)
            
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Descarga completada.")
        except Exception as e:
            print(f"❌ Error fatal descargando el modelo: {e}")
            raise e
    else:
        print("📂 El modelo ya existe localmente. Omitiendo descarga.")

@app.on_event("startup")
def load_model():
    """Ciclo de vida: Descarga -> Carga en Memoria"""
    global model
    
    # 1. Asegurar que el archivo exista (Descargar si es necesario)
    try:
        download_model()
    except Exception as e:
        print("⚠️ No se pudo descargar el modelo. La API funcionará pero fallará al predecir.")
        return

    # 2. Cargar en memoria con TensorFlow
    print(f"🔄 Cargando modelo en memoria desde: {LOCAL_MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        print("✅ ¡Modelo en memoria y listo para inferencia!")
    except Exception as e:
        print(f"❌ Error cargando .keras: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224)) 
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando imagen")

@app.get("/")
def home():
    return {"status": "online", "docs_url": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado o descargando...")
    
    contents = await file.read()
    processed_image = preprocess_image(contents)
    
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