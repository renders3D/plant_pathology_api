from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

client = TestClient(app)

def test_read_main():
    """Prueba que el endpoint raíz responda"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint_structure():
    """
    Prueba que el endpoint /predict acepte una imagen 
    y devuelva la estructura JSON correcta.
    (No validamos la predicción exacta porque el modelo puede variar,
     validamos que la API no explote).
    """
    # 1. Crear una imagen falsa en memoria (negra)
    file_bytes = io.BytesIO()
    image = Image.new('RGB', (256, 256), color='red')
    image.save(file_bytes, format='JPEG')
    file_bytes.seek(0) # Volver al inicio del archivo

    # 2. Enviar la imagen a la API
    # Usamos un nombre de archivo dummy
    files = {"file": ("test_image.jpg", file_bytes, "image/jpeg")}
    
    # Nota: Al usar TestClient, las BackgroundTasks se ejecutan síncronamente o se ignoran 
    # dependiendo de la config, pero no fallan si no hay credenciales de Azure reales en el test.
    response = client.post("/predict", files=files)
    
    # 3. Validaciones
    if response.status_code == 503:
        # Es aceptable que falle con 503 si el modelo no se descargó en el entorno de test
        # (GitHub Actions a veces no tiene conexión o tiempo para bajar 200MB en el test unitario)
        assert response.json()["detail"] == "Modelo no listo."
    else:
        # Si el modelo cargó (o si mockeamos), debe ser 200
        assert response.status_code == 200
        data = response.json()
        assert "deficiencia" in data
        assert "fusario" in data
        assert "sanas" in data
        # Verificar que los scores sean números
        assert isinstance(data["deficiencia"], float)
        assert isinstance(data["fusario"], float)
        assert isinstance(data["sanas"], float)

def test_preprocess_logic():
    """Prueba unitaria de la lógica de redimensionamiento"""
    from main import preprocess_image
    import numpy as np
    
    # Crear imagen gigante
    file_bytes = io.BytesIO()
    Image.new('RGB', (1000, 1000)).save(file_bytes, format='JPEG')
    file_bytes.seek(0)
    
    # Ejecutar preprocesamiento
    result = preprocess_image(file_bytes.read())
    
    # Validar que salga con el tamaño que espera el modelo (256x256 según tu último fix)
    assert result.shape == (1, 256, 256, 3)
    # Validar que esté normalizado (0-1)
    assert result.max() <= 1.0