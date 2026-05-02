import streamlit as st
import requests
import pandas as pd

# ================= CONFIG =================
API_URL = "https://your-fastapi-service.onrender.com"
API_KEY = "your_api_key"

headers = {"x-api-key": API_KEY}

st.set_page_config(layout="wide")
st.title("📊 AI Monitoring Dashboard")
st.caption("Real-time Model Observability 🚀")

# ================= FETCH =================
@st.cache_data(ttl=10)
def fetch_all():
    try:
        analytics = requests.get(f"{API_URL}/analytics", headers=headers).json()
        drift = requests.get(f"{API_URL}/drift", headers=headers).json()
        logs = requests.get(f"{API_URL}/logs", headers=headers).json()
        return analytics, drift, logs
    except:
        return None, None, None

analytics, drift, logs = fetch_all()

if not analytics:
    st.error("❌ API connection failed")
    st.stop()

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

col1.metric("Total Requests", analytics["total_requests"])
col2.metric("Avg Latency", round(analytics["avg_latency"], 4))
col3.metric("Avg Prediction", round(analytics["avg_prediction"], 4))

# ================= ALERT SYSTEM =================
st.subheader("🚨 System Alerts")

if analytics["avg_latency"] > 1:
    st.warning("⚠️ High latency detected!")

if drift["status"] == "drift detected":
    st.error(f"⚠️ Model Drift! Score: {drift['drift_score']:.2f}")
else:
    st.success("✅ Model Stable")

# ================= LOGS =================
st.subheader("📜 Recent Predictions")

df = pd.DataFrame(logs)

if not df.empty:
    st.dataframe(df)

    # ================= CHART =================
    st.subheader("📈 Latency Over Time")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    st.line_chart(df.set_index("timestamp")["latency"])

    # ================= PREDICTION DISTRIBUTION =================
    st.subheader("📊 Prediction Distribution")

    st.bar_chart(df["prediction"].value_counts())

# ================= REFRESH =================
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()