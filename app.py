import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Pro", layout="wide")
st.title("📊 Customer Segmentation & LTV Modeling App")

# --- Upload or Sample ---
st.sidebar.header("📁 Upload or Use Sample Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data uploaded.")
else:
    st.info("Using sample dataset for demonstration.")
    np.random.seed(42)
    df = pd.DataFrame({
        "CustomerID": range(1001, 1501),
        "Frequency": np.random.poisson(4, 500),
        "AvgSpend": np.random.normal(35, 10, 500).round(2),
        "Recency": np.random.randint(1, 90, 500),
        "Engagement": np.random.beta(2, 5, 500).round(2),
        "LoyaltyTier": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], 500, p=[0.3, 0.4, 0.2, 0.1]),
        "ChurnRisk": np.random.rand(500).round(2),
        "LTV": np.random.normal(400, 150, 500).round(2)
    })

# --- Standardize ID format ---
if "CustomerID" in df.columns:
    df["CustomerID"] = df["CustomerID"].astype(str)

# --- Preview ---
st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# --- EDA ---
st.sidebar.header("🔍 EDA")
if st.sidebar.checkbox("Show Summary Stats"):
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

if st.sidebar.checkbox("Show Null Values"):
    st.subheader("🚫 Missing Values")
    st.write(df.isnull().sum())

# --- Preprocessing ---
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
categorical_cols = [col for col in categorical_cols if col not in ["CustomerID"]]

if len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    df_encoded = df.copy()

features = df_encoded.drop(columns=["CustomerID", "LTV"], errors="ignore")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# --- Clustering ---
st.sidebar.header("🧠 Clustering Settings")
cluster_method = st.sidebar.radio("Clustering Algorithm", ["KMeans", "Gaussian Mixture"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

if cluster_method == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
else:
    model = GaussianMixture(n_components=n_clusters, random_state=42)

labels = model.fit_predict(X_scaled)
df["Segment"] = labels

# --- Silhouette Score ---
sil_score = silhouette_score(X_scaled, labels)
st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

# --- PCA Visualization ---
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Segment"] = labels

st.subheader("🧬 Cluster Visualization (PCA)")
fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Segment"].astype(str), title="Segment Clusters")
st.plotly_chart(fig, use_container_width=True)

# --- Segment Profiles ---
st.subheader("📌 Segment Profiles")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["LTV"], errors='ignore')
st.dataframe(df.groupby("Segment")[numeric_cols].mean().round(2))

# --- Segment Heatmap ---
st.subheader("🔥 Segment Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.groupby("Segment")[numeric_cols].mean(), cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
st.pyplot(fig)

# --- LTV Modeling ---
if "LTV" in df.columns:
    st.subheader("📈 Predictive Modeling: LTV")
    X_ltv = df_encoded.drop(columns=["CustomerID", "LTV"], errors="ignore")
    y_ltv = df_encoded["LTV"]
    model_ltv = LinearRegression()
    model_ltv.fit(X_ltv, y_ltv)
    df["LTV_Predicted"] = model_ltv.predict(X_ltv).round(2)
    st.dataframe(df[["CustomerID", "Segment", "LTV", "LTV_Predicted"]].head())

# --- Export ---
st.sidebar.header("📥 Export")
st.sidebar.download_button("Download CSV", data=df.to_csv(index=False).encode(), file_name="customer_segments_ltv.csv", mime="text/csv")
