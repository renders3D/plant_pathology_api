from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def test_read_main():
    """Prueba que el endpoint raíz responda"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint_structure():
    """
    Prueba que el endpoint /predict devuelva el nuevo formato JSON
    con probabilidades para las 3 clases.
    """
    # 1. Crear una imagen falsa (negra) de 224x224
    file_bytes = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='red')
    image.save(file_bytes, format='JPEG')
    file_bytes.seek(0)

    # 2. Enviar a la API
    files = {"file": ("test_image.jpg", file_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    # 3. Validaciones
    if response.status_code == 503:
        # Aceptable si el modelo no se ha descargado en el entorno CI
        assert response.json()["detail"] == "Modelo no listo."
    else:
        assert response.status_code == 200
        data = response.json()
        
        # Validar el nuevo formato (Diccionario de probabilidades)
        assert "deficiencia" in data
        assert "fusario" in data
        assert "sanas" in data
        
        # Validar que los valores sean floats
        assert isinstance(data["fusario"], float)

def test_preprocess_logic():
    """
    Prueba unitaria de la lógica de redimensionamiento para EfficientNet.
    Debe ser 224x224 y valores 0-255 (NO normalizados a 0-1).
    """
    from main import preprocess_image
    
    # Crear imagen gigante
    file_bytes = io.BytesIO()
    Image.new('RGB', (1000, 1000), color='white').save(file_bytes, format='JPEG')
    file_bytes.seek(0)
    
    # Ejecutar preprocesamiento
    result = preprocess_image(file_bytes.read())
    
    # 1. Validar tamaño (EfficientNet usa 224x224 nativo)
    assert result.shape == (1, 224, 224, 3)
    
    # 2. Validar rango de valores (EfficientNet espera 0-255)
    # Una imagen blanca ('white') tendrá valores cercanos a 255
    assert result.max() > 1.0 
    assert result.max() <= 255.0