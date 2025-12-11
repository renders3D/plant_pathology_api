from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import requests
import uuid
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient

app = FastAPI(
    title="Plant Pathology API (Final Production)",
    description="API TFLite Normalizada (0-1) y Redimensionada (256x256)",
    version="4.0.0"
)

# --- CONFIGURACIÓN ---
MODEL_FILENAME = "plant_model.tflite"
LOCAL_MODEL_PATH = f"models/{MODEL_FILENAME}"
MODEL_URL = "https://plantmodelsstorage.blob.core.windows.net/models/plant_model.tflite"  

# Azure Logging
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "inference-data"

# ✅ CONFIGURACIÓN GANADORA VALIDADA EN LABORATORIO
class_names = ['deficiencia', 'fusario', 'sanas'] 
TARGET_SIZE = (256, 256) # Validado por debug_keras.py

# Variables globales
interpreter = None
input_details = None
output_details = None
blob_service_client = None

def download_model():
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
    global interpreter, input_details, output_details, blob_service_client
    
    # 1. Cargar Modelo
    try:
        download_model()
        interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Motor TFLite listo. Input esperado: {input_details[0]['shape']}")
    except Exception as e:
        print(f"❌ Error TFLite: {e}")

    # 2. Conexión Azure
    try:
        if AZURE_CONNECTION_STRING:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            print("✅ Logging activado.")
        else:
            print("⚠️ Logging desactivado.")
    except Exception as e:
        print(f"❌ Error Azure: {e}")

def save_prediction_data(image_bytes, prediction_result, filename):
    if not blob_service_client: return
    try:
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        blob_client_img = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=f"{timestamp}_{request_id}.jpg")
        blob_client_img.upload_blob(image_bytes, overwrite=True)
        
        metadata = {
            "request_id": request_id,
            "timestamp": timestamp,
            "original_filename": filename,
            "prediction": prediction_result,
            "model_config": "256x256_Normalized_0_1"
        }
        blob_client_json = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=f"{timestamp}_{request_id}.json")
        blob_client_json.upload_blob(json.dumps(metadata), overwrite=True)
        print(f"💾 Log guardado: {request_id}")
    except Exception as e:
        print(f"⚠️ Error logging: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # ✅ APLICANDO LA FÓRMULA GANADORA (MÉTODO A)
        image = image.resize(TARGET_SIZE) 
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0  # Normalización 0-1
        
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando imagen")

@app.get("/")
def home():
    return {"status": "online", "model_version": "v4.0 (Fixed)", "docs_url": "/docs"}

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...), top_k: int = 3):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo no listo.")
    
    contents = await file.read()
    input_data = preprocess_image(contents)
    
    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predictions = output_data[0]

    # Asegurar top_k válido y generar resultados
    k = int(top_k)
    if k <= 0:
        k = 1
    k = min(k, predictions.shape[0])

    result_json = {}
    for i in range(0, k):
        label = class_names[i] if class_names and i < len(class_names) else f"Clase {i}"
        result_json[label] = float(predictions[i])

    # --- ENCOLAR TAREA DE LOGGING ---
    background_tasks.add_task(save_prediction_data, contents, result_json, file.filename)
    
    return result_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)