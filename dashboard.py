import streamlit as st
import requests
import pandas as pd

# ================= CONFIG =================
API_URL = "https://churning-prediction.onrender.com/"  # 🔴 CHANGE THIS
API_KEY = "helloworld"  # 🔴 CHANGE THIS

headers = {
    "x-api-key": API_KEY
}

# ================= PAGE =================
st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("📊 AI Model Monitoring Dashboard")
st.caption("Live AI Monitoring System 🚀")

# ================= FETCH DATA =================
@st.cache_data(ttl=10)
def fetch_data():
    try:
        analytics = requests.get(f"{API_URL}/analytics", headers=headers).json()
        drift = requests.get(f"{API_URL}/drift", headers=headers).json()
        return analytics, drift
    except Exception as e:
        return None, None

analytics, drift = fetch_data()

# ================= ERROR HANDLING =================
if not analytics or not drift:
    st.error("❌ Failed to fetch data from API")
    st.stop()

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

col1.metric("Total Requests", analytics.get("total_requests", 0))
col2.metric("Avg Latency", round(analytics.get("avg_latency", 0), 4))
col3.metric("Avg Prediction", round(analytics.get("avg_prediction", 0), 4))

# ================= DRIFT =================
st.subheader("📡 Model Drift Status")

if drift.get("status") == "drift detected":
    st.error(f"⚠️ Drift Detected (score: {drift.get('drift_score', 0):.2f})")
else:
    st.success("✅ Model Stable")

# ================= LATENCY BAR =================
st.subheader("⚡ Latency Overview")

latency = analytics.get("avg_latency", 0)
st.progress(min(latency, 1.0))

# ================= SIMPLE CHART =================
st.subheader("📈 Metrics Visualization")

data = pd.DataFrame({
    "Metric": ["Total Requests", "Avg Latency", "Avg Prediction"],
    "Value": [
        analytics.get("total_requests", 0),
        analytics.get("avg_latency", 0),
        analytics.get("avg_prediction", 0)
    ]
})

st.bar_chart(data.set_index("Metric"))

# ================= REFRESH =================
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()