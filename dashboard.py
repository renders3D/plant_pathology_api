import streamlit as st
import os
from azure.storage.blob import BlobServiceClient
import json
from PIL import Image
import io
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN ---
# Pega aquí tu Connection String (o usa os.getenv)
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "inference-data"

st.set_page_config(page_title="Plant Doctor Monitor", layout="wide")

@st.cache_resource
def get_blob_service():
    """Conecta a Azure una sola vez"""
    try:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    except Exception as e:
        st.error(f"Error conectando a Azure: {e}")
        return None

def get_recent_inferences(limit=10):
    """Descarga los últimos N archivos JSON y sus imágenes"""
    client = get_blob_service()
    if not client: return []
    
    container_client = client.get_container_client(CONTAINER_NAME)
    
    # Listar blobs y filtrar solo JSONs
    blobs = list(container_client.list_blobs())
    json_blobs = [b for b in blobs if b.name.endswith('.json')]
    
    # Ordenar por fecha (más reciente primero)
    json_blobs.sort(key=lambda x: x.creation_time, reverse=True)
    
    data = []
    
    # Barra de progreso visual
    progress_bar = st.progress(0)
    
    for i, blob in enumerate(json_blobs[:limit]):
        try:
            # 1. Descargar JSON
            bytes_json = container_client.download_blob(blob.name).readall()
            metadata = json.loads(bytes_json)
            
            prediction = ""
            confidence = 0.0
            for k, value in metadata.get("prediction").items():
                if value >= confidence:
                    prediction = k
                    confidence = value

            # 2. Descargar Imagen asociada (asumimos mismo nombre base .jpg)
            img_blob_name = blob.name.replace('.json', '.jpg')
            bytes_img = container_client.download_blob(img_blob_name).readall()
            
            # 3. Guardar en lista
            data.append({
                "timestamp": metadata.get("timestamp"),
                # "prediction": metadata.get("prediction", {}).get("prediction", "N/A"),
                # "confidence": metadata.get("prediction", {}).get("confidence", 0),
                "prediction": prediction,
                "confidence": confidence,
                "image_bytes": bytes_img,
                "filename": metadata.get("original_filename")
            })
            
        except Exception as e:
            print(f"Error procesando {blob.name}: {e}")
            
        # Actualizar barra
        progress_bar.progress((i + 1) / min(len(json_blobs), limit))
            
    progress_bar.empty()
    return data

# --- INTERFAZ GRÁFICA ---
st.title("🌿 Plant Doctor - MLOps Monitor")
st.markdown("Monitorización en tiempo real de inferencias en producción (Edge/Cloud).")

if st.button("🔄 Actualizar Datos"):
    st.cache_data.clear()

# Cargar datos
with st.spinner('Descargando datos desde Azure Blob Storage...'):
    inferences = get_recent_inferences(limit=12)

if not inferences:
    st.warning("No hay datos de inferencia aún. ¡Usa la API para generar predicciones!")
else:
    # Métricas Globales
    st.markdown("### 📊 Estadísticas Recientes")
    df = pd.DataFrame(inferences)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Inferencias", len(inferences))
    col2.metric("Confianza Promedio", f"{df['confidence'].mean():.1f}%")
    
    # Detección más común
    top_class = df['prediction'].mode()[0] if not df.empty else "N/A"
    col3.metric("Clase Frecuente", top_class)

    st.markdown("---")
    st.markdown("### 📸 Galería de Inferencia (Últimos Eventos)")

    # Grid de imágenes
    cols = st.columns(4) # 4 columnas
    for idx, item in enumerate(inferences):
        with cols[idx % 4]:
            try:
                # Mostrar imagen
                img = Image.open(io.BytesIO(item['image_bytes']))
                st.image(img, use_column_width=True)
                
                # Mostrar etiquetas
                color = "green" if item['confidence'] > 80 else "orange"
                st.markdown(f"**{item['prediction']}**")
                st.markdown(f":{color}[Confianza: {item['confidence']}%]")
                st.caption(f"📅 {item['timestamp'][:16].replace('T', ' ')}")
            except Exception as e:
                st.error("Img Error")

# Sidebar con información técnica
st.sidebar.header("🔧 MLOps Info")
st.sidebar.info("""
Este dashboard consume datos del contenedor **inference-data** en Azure.
Sirve para:
1. Detectar Data Drift.
2. Validar calidad del modelo.
3. Seleccionar imágenes para re-entrenamiento.
""")