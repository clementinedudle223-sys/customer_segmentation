# Advanced Customer Segmentation App (Streamlit)
# Features: CSV Upload, Multiple Clustering Options, PCA Plot, Segment Summary, PDF Report Export

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="Advanced Customer Segmentation", layout="wide")
st.title("🧠 Advanced Customer Segmentation App")

# --- Upload Data ---
st.sidebar.header("1. Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Sample dataset if no file uploaded
def load_sample_data():
    np.random.seed(42)
    return pd.DataFrame({
        "frequency": np.random.poisson(3, 500),
        "avg_order_value": np.random.normal(25, 8, 500),
        "loyalty_score": np.random.uniform(0, 1, 500),
        "engagement_score": np.random.normal(0.5, 0.2, 500)
    })

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    st.sidebar.info("Using sample dataset")
    df = load_sample_data()

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- Preprocessing ---
X = df.select_dtypes(include=np.number)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Clustering ---
st.sidebar.header("2. Choose Clustering Method")
method = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

if method == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif method == "DBSCAN":
    eps = st.sidebar.slider("EPS (Neighborhood Radius)", 0.1, 3.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif method == "Agglomerative":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    model = AgglomerativeClustering(n_clusters=n_clusters)

clusters = model.fit_predict(X_scaled)
df["Segment"] = clusters

# --- PCA Plot ---
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
df_pca["Segment"] = clusters

st.subheader("🧬 Cluster Visualization (PCA)")
fig = px.scatter(df_pca, x="PC1", y="PC2", color=df_pca["Segment"].astype(str), title="Customer Segments (PCA)", opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

# --- Segment Summary ---
st.subheader("📌 Segment Profiles")
summary = df.groupby("Segment").agg(["mean", "count"])
st.dataframe(summary)

# --- Heatmap ---
st.subheader("🔥 Segment Heatmap")
heat_data = df.groupby("Segment").mean(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Download Report (CSV for now) ---
st.sidebar.header("3. Export")
buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
st.sidebar.download_button("Download Segmented Data", data=buf, file_name="segmented_customers.csv", mime="text/csv")
