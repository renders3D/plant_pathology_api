from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import requests
import uuid
from datetime import datetime
import json
from azure.storage.blob import BlobServiceClient

app = FastAPI(
    title="Plant Pathology TFLite API",
    description="API Optimizada (Edge-Ready) para detección de patologías",
    version="2.0.0"
)

# --- CONFIGURACIÓN TFLITE ---
MODEL_FILENAME = "plant_model.tflite"
LOCAL_MODEL_PATH = f"models/{MODEL_FILENAME}"
MODEL_URL = "https://plantmodelsstorage.blob.core.windows.net/models/plant_model.tflite" 

class_names = ['deficiencia', 'fusario', 'sanas'] 

# --- CONFIGURACIÓN DE LOGGING (NUEVO) ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING") 
CONTAINER_NAME = "inference-data"

# Variables globales
interpreter = None
input_details = None
output_details = None
blob_service_client = None # Cliente para subir datos

def download_model():
    """Descarga el modelo ligero desde Azure"""
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"⬇️ Descargando modelo: {MODEL_URL}")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Descarga completada.")
        except Exception as e:
            print(f"❌ Error descargando: {e}")

@app.on_event("startup")
def startup_event():
    """Carga modelo y conexión a Azure"""
    global interpreter, input_details, output_details, blob_service_client
    
    # 1. Cargar Modelo
    try:
        download_model()
        print(f"⚡ Iniciando TFLite...")
        interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Motor TFLite listo.")
    except Exception as e:
        print(f"❌ Error TFLite: {e}")

    # 2. Conectar a Azure Storage (Para guardar datos)
    try:
        if "DefaultEndpointsProtocol" in AZURE_CONNECTION_STRING:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            print("✅ Conectado a Azure Blob Storage para Logging.")
        else:
            print("⚠️ No se configuró Connection String. El logging estará desactivado.")
    except Exception as e:
        print(f"❌ Error conectando a Azure Storage: {e}")

def save_prediction_data(image_bytes, prediction_result, filename):
    """
    TAREA EN SEGUNDO PLANO:
    Sube la imagen y el JSON del resultado a Azure para futuro reentrenamiento.
    """
    if not blob_service_client:
        return

    try:
        # Generar ID único para este evento
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Nombres de archivo
        img_blob_name = f"{timestamp}_{request_id}.jpg"
        json_blob_name = f"{timestamp}_{request_id}.json"
        
        # 1. Subir Imagen
        blob_client_img = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=img_blob_name)
        blob_client_img.upload_blob(image_bytes, overwrite=True)
        
        # 2. Preparar y Subir Metadatos (JSON)
        metadata = {
            "request_id": request_id,
            "timestamp": timestamp,
            "original_filename": filename,
            "prediction": prediction_result,
            "model_version": "3.0.0 (TFLite)"
        }
        blob_client_json = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=json_blob_name)
        blob_client_json.upload_blob(json.dumps(metadata), overwrite=True)
        
        print(f"💾 [LOG] Datos guardados en Azure: {request_id}")
        
    except Exception as e:
        print(f"⚠️ Error guardando datos de inferencia: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((256, 256)) 
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando imagen")

@app.get("/")
def home():
    return {"status": "online", "docs_url": "/docs"}

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo no listo.")
    
    # Leer archivo
    contents = await file.read()
    
    # Preprocesar
    input_data = preprocess_image(contents)
    
    # Inferencia TFLite
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predictions = output_data[0]
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    result_class = class_names[class_idx]

    result_json = {
        "prediction": result_class,
        "confidence": round(confidence * 100, 2),
        "engine": "TFLite Edge"
    }

    # --- MAGIA MLOps ---
    # Enviamos la tarea de guardar a segundo plano.
    # La API responde al usuario INMEDIATAMENTE, y luego sube la foto.
    background_tasks.add_task(save_prediction_data, contents, result_json, file.filename)
    # -------------------

    return result_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)