import streamlit as st
import psycopg2
import pandas as pd
import os

# ================= DB CONNECTION =================
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("📊 AI MODEL DASHBOARD")

# ================= LOAD DATA =================
cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100")
rows = cur.fetchall()

df = pd.DataFrame(rows, columns=[
    "id", "input", "prediction", "latency", "model_version", "timestamp"
])

# ================= METRICS =================
cur.execute("SELECT COUNT(*) FROM predictions")
total = cur.fetchone()[0]

cur.execute("SELECT AVG(latency) FROM predictions")
avg_latency = cur.fetchone()[0]

cur.execute("SELECT AVG(prediction) FROM predictions")
avg_prediction = cur.fetchone()[0]

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)
col2.metric("Avg Latency (s)", round(avg_latency or 0, 4))
col3.metric("Avg Prediction Rate", round(avg_prediction or 0, 4))

st.divider()

# ================= DRIFT =================
st.subheader("📉 Model Drift Insight")

drift = abs((avg_prediction or 0.5) - 0.5)

if drift > 0.2:
    st.error(f"Drift Detected 🚨 (score: {drift:.3f})")
else:
    st.success(f"Model Stable ✅ (score: {drift:.3f})")

st.divider()

# ================= TABLE =================
st.subheader("📦 Recent Predictions")

st.dataframe(df)