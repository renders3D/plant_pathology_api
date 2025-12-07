import tensorflow as tf
import os

# 1. Configuración
INPUT_MODEL_PATH = "models/plant_model.keras"  # Tu modelo actual
OUTPUT_MODEL_PATH = "models/plant_model.tflite"

def convert():
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"❌ Error: No encuentro el archivo {INPUT_MODEL_PATH}")
        return

    print(f"🔄 Cargando modelo Keras desde {INPUT_MODEL_PATH}...")
    # Cargar el modelo original
    model = tf.keras.models.load_model(INPUT_MODEL_PATH)

    # 2. Convertir a TFLite
    print("⚙️ Iniciando conversión a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Opcional: Optimizaciones extra (Descomentar para reducir aún más el tamaño si es necesario)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    tflite_model = converter.convert()

    # 3. Guardar
    print(f"💾 Guardando modelo convertido en {OUTPUT_MODEL_PATH}...")
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    # 4. Comparar tamaños
    size_keras = os.path.getsize(INPUT_MODEL_PATH) / (1024 * 1024)
    size_tflite = os.path.getsize(OUTPUT_MODEL_PATH) / (1024 * 1024)
    
    print(f"\n✅ ¡ÉXITO! Resultados:")
    print(f"   - Modelo Original: {size_keras:.2f} MB")
    print(f"   - Modelo TFLite:   {size_tflite:.2f} MB")
    print(f"   - Reducción:       {100 - (size_tflite/size_keras)*100:.1f}%")

if __name__ == "__main__":
    convert()
    