import streamlit as st
import pandas as pd
import requests
import sqlite3
import time
# pyrefly: ignore [missing-import]
import plotly.express as px
from datetime import datetime
import pytz

# Configure page layout
st.set_page_config(page_title="SOC Dashboard - NIDS", layout="wide", page_icon="🛡️")

import os

# API Base URL (Use 'api' as hostname in Docker, but default to localhost for native testing)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_db_connection():
    return sqlite3.connect('./data/nids.db', check_same_thread=False)

def fetch_predictions():
    """Fetches the latest predictions from the database and converts time to IST."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1000", conn)
        if not df.empty:
            # Convert string UTC timestamp to datetime, then to IST
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Localize to UTC if naive, then convert to Asia/Kolkata (IST)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
        return df
    except Exception as e:
        return pd.DataFrame()
    finally:
        conn.close()

# Sidebar Settings
st.sidebar.title("🛡️ Settings Panel")
st.sidebar.markdown("Configure your Network Intrusion Detection System.")

# Fetch available interfaces
try:
    response = requests.get(f"{API_BASE_URL}/interfaces")
    interfaces = response.json()
    iface_options = list(interfaces.keys())
except Exception:
    iface_options = ["None Detected"]

st.sidebar.subheader("Live Network Capture")
selected_iface = st.sidebar.selectbox("Network Interface", iface_options)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Start Monitoring"):
        try:
            res = requests.post(f"{API_BASE_URL}/monitor/start?interface={selected_iface}")
            st.sidebar.success(res.json().get("message", "Started"))
        except:
            st.sidebar.error("API Unreachable")
            
with col2:
    if st.button("Stop Monitoring"):
        try:
            res = requests.post(f"{API_BASE_URL}/monitor/stop")
            st.sidebar.success(res.json().get("message", "Stopped"))
        except:
            st.sidebar.error("API Unreachable")

st.sidebar.markdown("---")
st.sidebar.subheader(" Interactive Demo Mode")
st.sidebar.markdown("Simulate live traffic by streaming local PCAP/CSV data directly into the AI Pipeline.")
demo1, demo2 = st.sidebar.columns(2)
with demo1:
    if st.button("Start Simulation"):
        try:
            res = requests.post(f"{API_BASE_URL}/simulate/start")
            st.sidebar.success("Simulation Started")
        except:
            st.sidebar.error("API Unreachable")
with demo2:
    if st.button("Stop Simulation"):
        try:
            res = requests.post(f"{API_BASE_URL}/simulate/stop")
            st.sidebar.success("Simulation Stopped")
        except:
            st.sidebar.error("API Unreachable")

st.sidebar.markdown("---")
st.sidebar.subheader("Report Generation")
if st.sidebar.button("Generate CSV Report"):
    df = fetch_predictions()
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Latest Data (CSV)",
        data=csv,
        file_name=f'nids_report_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# Main Dashboard Layout
st.title("Network Intrusion Detection Center")

@st.fragment(run_every="2s")
def render_dashboard():
    # Fetch latest data
    df = fetch_predictions()

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🔴 Live Monitoring", "📜 Detailed Alert Logs", "🏗️ Project Workflow & Architecture"])

    with tab1:
        if df.empty:
            st.info("No network traffic captured yet. Start Monitoring or click **Start Simulation** in the sidebar to see live data.")
        else:
            # High level metrics
            total_packets = len(df)
            attacks = len(df[df['severity'] != 'Low'])
            critical_attacks = len(df[df['severity'] == 'Critical'])
            avg_latency = df['latency_ms'].mean()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Flows Analysed", total_packets)
            m2.metric("Threats Detected", attacks, f"{(attacks/total_packets*100):.1f}%" if total_packets > 0 else "0%")
            m3.metric("Critical Alerts", critical_attacks)
            m4.metric("Avg Inference Latency", f"{avg_latency:.2f} ms")
            
            st.markdown("---")
            
            # Layout columns
            col1, col2 = st.columns((2, 1))
            
            with col1:
                st.subheader("Live Attack Timeline (IST)")
                # Group by 10-second intervals for a faster moving timeline
                df['time_window'] = df['timestamp'].dt.floor('10S')
                timeline = df.groupby(['time_window', 'severity']).size().reset_index(name='count')
                fig = px.area(timeline, x="time_window", y="count", color="severity", 
                              color_discrete_map={"Low": "green", "Medium": "yellow", "High": "orange", "Critical": "red"})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Threat Summary")
                threat_counts = df['prediction'].value_counts().reset_index()
                threat_counts.columns = ['Attack Type', 'Count']
                fig2 = px.pie(threat_counts, values='Count', names='Attack Type', hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("---")
            st.subheader("Active Critical & High Alerts")
            alerts_df = df[df['severity'].isin(['Critical', 'High'])].head(10).copy()
            if not alerts_df.empty:
                alerts_df['timestamp'] = alerts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(
                    alerts_df[['timestamp', 'source_ip', 'destination_ip', 'prediction', 'confidence', 'severity', 'top_shap_feature']],
                    use_container_width=True
                )
            else:
                st.success("No active Critical or High alerts!")

    with tab2:
        st.subheader("Detailed Network Logs")
        st.markdown("View all captured network flows and predictions. Time displayed in Indian Standard Time (IST).")
        if not df.empty:
            display_df = df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_df, use_container_width=True, height=600)
        else:
            st.info("No data available.")

    with tab3:
        st.subheader("Project Workflow & Architecture")
        st.markdown("""
        This Network Intrusion Detection System (NIDS) is a complete, end-to-end Machine Learning Operations (MLOps) pipeline designed for real-time cybersecurity.

        ### 1. Data Ingestion & Validation
        Raw PCAP (Packet Capture) files are ingested, converted into bidirectional flow features, and passed into our strict **Data Validation Pipeline**. A Pydantic and YAML-driven schema enforces that all 77 required features exist, handling missing data intelligently to prevent API crashes.

        ### 2. LightGBM AI Model
        The core brain is a highly optimized **LightGBM Classifier**, trained on the CIC-IDS2017 dataset. It was selected for its ultra-fast inference speed and high accuracy on tabular network data. The model categorizes traffic into 15 unique classes (e.g., BENIGN, DoS Hulk, PortScan, DDoS).

        ### 3. Real-Time Threat Engine
        Raw predictions are piped into a custom **Threat Engine**, which dynamically assigns SOC-standard severity labels:
        - **Low**: Benign Traffic
        - **Medium**: Infiltration / Web Attacks (low confidence)
        - **High**: PortScans, FTP-Patator
        - **Critical**: High-confidence DDoS and DoS Hulk attacks

        ### 4. Explainable AI (SHAP)
        To break the "Black Box" of Machine Learning, we integrated **SHAP (SHapley Additive exPlanations)**. Whenever a High or Critical alert is triggered, the AI computes exactly *which network feature* (e.g., `Bwd Packet Length Std`) caused the alert, giving security analysts actionable context.

        ### 5. Deployment Architecture
        - **FastAPI**: Serves the asynchronous backend API.
        - **SQLite & SQLAlchemy**: Persists all predictions instantly for querying.
        - **NFStream**: Powers the live packet sniffer on the host interface.
        - **Docker**: The entire stack is containerized using `docker-compose`, making deployment instantaneous across any environment.
        """)

# Render the fragment
render_dashboard()
