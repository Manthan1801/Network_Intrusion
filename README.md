#  AI-Powered Network Intrusion Detection System (NIDS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-orange.svg)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) pipeline built to detect, classify, and explain network cyberattacks in real-time. 

##  Key Features

- **Blazing Fast AI**: Powered by a highly optimized **LightGBM Classifier**, trained on the CIC-IDS2017 dataset, capable of categorizing live network traffic into 15 distinct classes (e.g., DoS Hulk, DDoS, PortScan, BENIGN).
- **Explainable AI (SHAP)**: Doesn't just flag an attack—it explains *why*. Our SHAP integration breaks open the black-box model to tell security analysts exactly which network feature triggered a Critical alert.
- **Dynamic Threat Engine**: Automatically maps raw model confidence into SOC-standard severities (Low, Medium, High, Critical).
- **Interactive SOC Dashboard**: A beautiful, real-time Security Operations Center (SOC) built in Streamlit, featuring live attack timelines, threat distribution pie charts, and auto-refreshing detailed logs.
- **Live Traffic Simulator**: Includes a powerful backend simulation engine that streams test PCAP/CSV data dynamically to visualize a live cyberattack without configuring network taps.
- **Production-Ready**: fully containerized with `docker-compose` and driven by a robust asynchronous FastAPI backend backed by SQLite/SQLAlchemy.

##  Architecture

1. **FastAPI Backend (`app/main.py`)**: The asynchronous core handling inference requests.
2. **Data Validation**: Strict Pydantic and YAML-driven validation ensuring data integrity before model inference.
3. **ML Pipeline (`core/`)**: Loads the serialized LightGBM model, preprocessing scalers, and label encoders.
4. **Streamlit UI (`dashboard/app.py`)**: The frontend interface serving real-time analytics to the user.

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
