from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf # Usaremos tf.lite
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import requests

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

# Variables globales para el Interprete
interpreter = None
input_details = None
output_details = None

def download_model():
    """Descarga el modelo ligero desde Azure"""
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"⬇️ Descargando modelo TFLite optimizado: {MODEL_URL}")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Descarga completada.")
        except Exception as e:
            print(f"❌ Error descargando: {e}")
            raise e

@app.on_event("startup")
def load_model():
    """Carga el Interprete TFLite (Consumo de RAM mínimo)"""
    global interpreter, input_details, output_details
    
    try:
        download_model()
        print(f"⚡ Iniciando Interprete TFLite con: {LOCAL_MODEL_PATH}")
        
        # Cargar el interprete
        interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL_PATH)
        interpreter.allocate_tensors() # Reserva SOLO la memoria necesaria
        
        # Obtener referencias a los tensores de entrada y salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ ¡Motor de inferencia TFLite listo!")
        
    except Exception as e:
        print(f"❌ Error fatal iniciando TFLite: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((256, 256)) 
        
        # Normalización (igual que antes)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando imagen")

@app.get("/")
def home():
    return {"status": "online", "engine": "TFLite", "docs_url": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="El cerebro TFLite no está listo.")
    
    contents = await file.read()
    input_data = preprocess_image(contents)
    
    # --- INFERENCIA CON TFLITE (El cambio manual) ---
    
    # 1. Poner los datos en el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 2. Ejecutar la inferencia (Invoke)
    interpreter.invoke()
    
    # 3. Leer los datos del tensor de salida
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Procesar resultados (igual que antes)
    predictions = output_data[0]
    # Aplicar Softmax si el TFLite no lo incluye (a veces TFLite saca logits crudos)
    # Si los valores no suman 1, descomentar la siguiente línea:
    # predictions = tf.nn.softmax(predictions).numpy()
    
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    result_class = class_names[class_idx]

    return {
        "prediction": result_class,
        "confidence": round(confidence * 100, 2),
        "engine": "TFLite Edge"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)