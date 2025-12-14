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
# Usa variables de entorno o pega tu string aquí para local
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "inference-data"

st.set_page_config(page_title="Plant Doctor Monitor v2", layout="wide", page_icon="🌿")

@st.cache_resource
def get_blob_service():
    try:
        if "DefaultEndpointsProtocol" not in AZURE_CONNECTION_STRING:
            return None
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    except Exception as e:
        return None

def get_recent_inferences(limit=20):
    client = get_blob_service()
    if not client: return []
    
    container_client = client.get_container_client(CONTAINER_NAME)
    blobs = list(container_client.list_blobs())
    json_blobs = [b for b in blobs if b.name.endswith('.json')]
    json_blobs.sort(key=lambda x: x.creation_time, reverse=True)
    
    data = []
    # Barra de progreso
    progress_bar = st.progress(0)
    
    for i, blob in enumerate(json_blobs[:limit]):
        try:
            bytes_json = container_client.download_blob(blob.name).readall()
            metadata = json.loads(bytes_json)
            
            img_blob_name = blob.name.replace('.json', '.jpg')
            bytes_img = container_client.download_blob(img_blob_name).readall()
            
            # --- LÓGICA DE PARSEO INTELIGENTE (Soporta v3 y v5) ---
            raw_preds = metadata.get("predictions", None)
            
            if raw_preds:
                # FORMATO NUEVO (Diccionario: {'fusario': 0.8, ...})
                # Encontrar la clase ganadora
                top_class = max(raw_preds, key=raw_preds.get)
                confidence = raw_preds[top_class] * 100
                detailed_probs = raw_preds # Guardamos todo para el gráfico
            else:
                # FORMATO ANTIGUO
                old_pred_struct = metadata.get("prediction", {})
                top_class = old_pred_struct.get("prediction", "N/A")
                confidence = old_pred_struct.get("confidence", 0)
                detailed_probs = None

            data.append({
                "timestamp": metadata.get("timestamp"),
                "prediction": top_class,
                "confidence": confidence,
                "detailed_probs": detailed_probs,
                "model_version": metadata.get("model_version", "Legacy"),
                "image_bytes": bytes_img,
                "filename": metadata.get("original_filename", "unknown")
            })
            
        except Exception:
            pass
        progress_bar.progress((i + 1) / min(len(json_blobs), limit))
            
    progress_bar.empty()
    return data

# --- UI ---
st.title("🌿 Plant Doctor AI - Production Monitor")
st.markdown(f"**Status:** Connected to Azure Blob Storage | **Container:** `{CONTAINER_NAME}`")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

if not AZURE_CONNECTION_STRING or "TU_CONNECTION" in AZURE_CONNECTION_STRING:
    st.error("⚠️ Please configure your AZURE_STORAGE_CONNECTION_STRING in the code.")
else:
    with st.spinner('Fetching telemetry from the cloud...'):
        inferences = get_recent_inferences(limit=12)

    if not inferences:
        st.info("No inference data found yet. Start predicting!")
    else:
        # Métricas
        df = pd.DataFrame(inferences)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inferences", len(inferences))
        c2.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
        c3.metric("Top Detected Class", df['prediction'].mode()[0] if not df.empty else "N/A")
        c4.metric("Latest Model", df.iloc[0]['model_version'])

        st.divider()
        st.subheader("📸 Live Inference Stream")

        # Grid Layout
        cols = st.columns(3)
        for idx, item in enumerate(inferences):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Imagen
                    img = Image.open(io.BytesIO(item['image_bytes']))
                    st.image(img, use_container_width=True)
                    
                    # Encabezado
                    color = "green" if item['confidence'] > 80 else "orange" if item['confidence'] > 50 else "red"
                    st.markdown(f"### :{color}[{item['prediction'].upper()}]")
                    st.caption(f"Confidence: **{item['confidence']:.1f}%** | {item['timestamp'][:16].replace('T', ' ')}")
                    
                    # Gráfico de Barras Mini (Si es modelo nuevo)
                    if item['detailed_probs']:
                        probs_df = pd.DataFrame(
                            list(item['detailed_probs'].items()), 
                            columns=['Class', 'Probability']
                        )
                        st.bar_chart(probs_df.set_index('Class'), height=100)
                    else:
                        st.info("Legacy Model Data (No detailed probs)")