# 🌱 Plant Pathology Inference API (Edge-Ready)

![Azure](https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

A production-grade, containerized Inference API designed to classify plant pathologies (*Fusarium, Deficiency, Healthy*) using Computer Vision. 

Optimized for **Edge Computing** (Drones/IoT) using TFLite and deployed on **Azure Cloud** with a robust CI/CD pipeline.

## 🚀 Key Features

* **⚡ High Performance:** Uses **EfficientNetB0** (optimized to `.tflite`) for sub-second inference on CPU.
* **☁️ Cloud-Native:** Fully Dockerized and deployed on Azure Web Apps for Containers.
* **🔄 Data Flywheel:** Asynchronous data logging to Azure Blob Storage for future model retraining (Observability).
* **🛡️ Robust CI/CD:** GitHub Actions pipeline with automated Unit Tests (`pytest`) and Multi-Architecture Build (AMD64/ARM64).

## 🏗️ Architecture

1.  **Input:** Image sent via REST API (`POST /predict`).
2.  **Processing:** Image resizing (224x224) and preprocessing (0-255 range).
3.  **Inference:** TFLite Interpreter runs the quantized EfficientNetB0 model.
4.  **Output:** JSON response with full probability distribution.
5.  **Telemetry (Async):** Image and result are uploaded to Azure Blob Storage in the background without blocking the response.

## 🛠️ Tech Stack

* **Framework:** FastAPI + Uvicorn
* **ML Engine:** TensorFlow Lite (TFLite) Runtime
* **Containerization:** Docker (Python 3.9 Slim)
* **Cloud:** Azure Web App, Azure Container Registry (ACR), Azure Blob Storage.
* **Testing:** Pytest, HTTPX

## 📦 Installation & Local Run

### Prerequisites
* Docker installed
* Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/renders3D/plant_pathology_api.git](https://github.com/renders3D/plant_pathology_api.git)
    cd PlantPathology_API
    ```

2.  **Build the Docker Image:**
    ```bash
    docker build -t plant-api .
    ```

3.  **Run the Container:**
    ```bash
    docker run -p 80:80 -e AZURE_STORAGE_CONNECTION_STRING="your_string" plant-api
    ```

4.  **Access Swagger UI:**
    Open `http://localhost/docs` in your browser.

## 🧪 Testing

Run the automated test suite to validate the API logic and image preprocessing:

```bash
pip install -r requirements.txt
pytest test_main.py