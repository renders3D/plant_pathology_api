# Usamos una imagen base oficial de Python ligera
FROM python:3.9-slim

# Evita que Python genere archivos .pyc y fuerza salida en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo en el contenedor
WORKDIR /app

# --- CORRECCIÓN IMPORTANTE ---
# Instalamos 'build-essential' (compiladores) además de las librerías gráficas.
# Esto evita que pip falle si necesita compilar algo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
# Aumentamos el timeout por si TensorFlow tarda en bajar
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copiar el código fuente y la carpeta models
COPY . .

# Exponer el puerto 8000
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
