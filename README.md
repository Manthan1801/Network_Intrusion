#  AI-Powered Network Intrusion Detection System (NIDS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-orange.svg)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) pipeline built to detect, classify, and explain network cyberattacks in real-time. 

##  Key Features

- **Comprehensive Training & Dataset**: Trained on the massive **[CIC-IDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)**, which contains over 2.8 million network traffic flows. The data underwent rigorous cleaning, handling of class imbalances, and feature scaling to ensure the model accurately captures real-world attack signatures while maintaining an extremely low false-positive rate.
- **Blazing Fast AI**: Powered by a highly optimized **LightGBM Classifier**, capable of categorizing live network traffic into 15 distinct classes (e.g., DoS Hulk, DDoS, PortScan, BENIGN).
- **Explainable AI (SHAP)**: Doesn't just flag an attack—it explains *why*. Our SHAP integration breaks open the black-box model to tell security analysts exactly which network feature triggered a Critical alert.
- **Dynamic Threat Engine**: Automatically maps raw model confidence into SOC-standard severities (Low, Medium, High, Critical).
- **Interactive SOC Dashboard (Testing Interface)**: While the core system is designed as a headless backend engine for enterprise environments, a beautiful, real-time Security Operations Center (SOC) built in Streamlit is included for visualization, testing, and understanding how the internal ML pipeline processes traffic.
- **Live Traffic Simulator**: Includes a powerful backend simulation engine that streams test PCAP/CSV data dynamically to visualize a live cyberattack without configuring network taps.
- **Production-Ready**: fully containerized with `docker-compose` and driven by a robust asynchronous FastAPI backend backed by SQLite/SQLAlchemy.

##  Architecture & Machine Learning Pipeline

In a true industry environment, this system operates entirely headless, processing millions of packets in the background. The architecture is split into several crucial stages:

1. **Data Ingestion & Validation (`core/validation.py`)**: Raw network features flow into the system and pass through a strict Pydantic and YAML-driven schema. This guarantees data integrity, intelligently handling missing values before inference to prevent critical API crashes.
2. **Preprocessing Layer (`core/` pipelines)**: The validated data is transformed using pre-fitted `RobustScaler` and `StandardScaler` artifacts, mapping real-time live capture data into the exact format the ML model expects.
3. **Model Inference**: The highly-tuned LightGBM model processes the scaled array and outputs a probability array across all 15 attack classes.
4. **Threat Engine (`core/threat_engine.py`)**: The raw AI predictions are too granular for human operators. The Threat Engine maps these outputs into SOC-standard severity levels (Low, Medium, High, Critical) based on class confidence.
5. **FastAPI Backend (`app/main.py`)**: The asynchronous core that orchestrates this entire pipeline, serving real-time requests and logging all predictions to the SQLite/SQLAlchemy database.
6. **Streamlit UI (`dashboard/app.py`)**: An attached frontend interface purely to demonstrate the internal ML engine's speed and accuracy visually.

##  Quick Start (Docker)

The absolute easiest way to run the entire architecture (Frontend + Backend + Database) is via Docker.

```bash
# Clone the repository
git clone https://github.com/Manthan1801/Network_Intrusion.git
cd Network_Intrusion

# Build and start the containers
docker-compose up --build
```
Once running, open your browser:
- **SOC Dashboard**: `http://localhost:8501`
- **API Swagger UI**: `http://localhost:8000/docs`

##  Manual Setup (Local Development)

If you prefer to run it natively without Docker:

1. **Install Dependencies**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # On Windows
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Backend**
   ```bash
   uvicorn app.main:app
   ```

3. **Start the Streamlit Dashboard**
   Open a *new* terminal and run:
   ```bash
   streamlit run dashboard/app.py
   ```
##  Running the Live Simulation

To see the dashboard spring to life without live network traffic:
1. Open the **SOC Dashboard** (`http://localhost:8501`).
2. Open the sidebar on the left.
3. Click the **Start Simulation** button under *Interactive Demo Mode*.
4. Watch the charts dynamically update in real-time as packets are processed!

##  Author
**Manthan Jikadara**

##  License
This project is licensed under the terms of the license included in this repository.
