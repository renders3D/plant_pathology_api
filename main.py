from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(
    title="Plant Pathology Inference API",
    description="API para clasificación de patologías en cultivos vía Drone/App",
    version="1.0.0"
)

# --- CONFIGURACIÓN ---
MODEL_PATH = "model/plant_model.h5"
class_names = ['deficiencia', 'fusario', 'sanas']

# Variable global para el modelo
model = None

@app.on_event("startup")
def load_model():
    """Carga el modelo en memoria al iniciar el contenedor"""
    global model
    print(f"Cargando modelo desde {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"Error fatal cargando el modelo: {e}")

def preprocess_image(image_bytes):
    """Transforma los bytes de la imagen al formato que espera la CNN"""
    try:
        # 1. Leer imagen
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Redimensionar
        image = image.resize((256, 256))
        
        # 3. Convertir a array y normalizar (si se entrenó con /255.0)
        img_array = np.array(image) / 255.0
        
        # 4. Expandir dimensiones (batch size de 1) -> (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando la imagen. Asegúrate que sea un archivo válido.")

@app.get("/")
def home():
    return {"status": "online", "service": "Plant Pathology Detector"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint principal: Recibe imagen, devuelve patología"""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado aún.")
    
    # Leer contenido del archivo
    contents = await file.read()
    
    # Preprocesar
    processed_image = preprocess_image(contents)
    
    # Inferencia
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    result_class = class_names[class_idx]

    return {
        "prediction": result_class,
        "confidence": round(confidence * 100, 2), # Porcentaje
        "metadata": {
            "filename": file.filename,
            "content_type": file.content_type
        }
    }

if __name__ == "__main__":
    # Esto permite correrlo localmente sin Docker para probar
    uvicorn.run(app, host="0.0.0.0", port=8000)