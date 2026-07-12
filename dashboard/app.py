import streamlit as st
import pandas as pd
import requests
import sqlite3
import time
# pyrefly: ignore [missing-import]
import plotly.express as px
from datetime import datetime

# Configure page layout
st.set_page_config(page_title="SOC Dashboard - NIDS", layout="wide", page_icon="🛡️")

# API Base URL (Use 'api' as the hostname because Docker Compose resolves service names)
API_BASE_URL = "http://api:8000"

def get_db_connection():
    return sqlite3.connect('./data/nids.db', check_same_thread=False)

def fetch_predictions():
    """Fetches the latest predictions from the database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1000", conn)
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

# Fetch latest data
df = fetch_predictions()

if df.empty:
    st.info("No network traffic captured yet. Start monitoring to see live data.")
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
        st.subheader("Live Attack Timeline")
        # Group by minute and severity
        df['minute'] = pd.to_datetime(df['timestamp']).dt.floor('Min')
        timeline = df.groupby(['minute', 'severity']).size().reset_index(name='count')
        fig = px.area(timeline, x="minute", y="count", color="severity", 
                      color_discrete_map={"Low": "green", "Medium": "yellow", "High": "orange", "Critical": "red"})
        st.plotly_chart(fig, use_container_width=True, key="timeline_chart")
        
    with col2:
        st.subheader("Threat Summary")
        threat_counts = df['prediction'].value_counts().reset_index()
        threat_counts.columns = ['Attack Type', 'Count']
        fig2 = px.pie(threat_counts, values='Count', names='Attack Type', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True, key="threat_chart")
        
    st.markdown("---")
    st.subheader("Active Critical & High Alerts")
    alerts_df = df[df['severity'].isin(['Critical', 'High'])].head(10)
    if not alerts_df.empty:
        st.dataframe(
            alerts_df[['timestamp', 'source_ip', 'destination_ip', 'prediction', 'confidence', 'severity', 'top_shap_feature']],
            use_container_width=True
        )
    else:
        st.success("No active Critical or High alerts!")
        
# Auto-refresh mechanism
time.sleep(2)
st.rerun()
