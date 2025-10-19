
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Demo", layout="wide")

st.title("🔍 Customer Segmentation – Mock Demo")
st.markdown("This is a mock segmentation project for a fictional meal delivery company. The app demonstrates how to use clustering to identify key customer personas.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("mock_customer_data.csv")

df = load_data()
st.subheader("📊 Raw Customer Data")
st.dataframe(df.head())

# Preprocessing
features = ['purchase_frequency', 'avg_order_value', 'loyalty_score', 'engagement_score']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster profiles
cluster_summary = df.groupby('cluster')[features].mean()

st.subheader("🧠 Cluster Profiles")
st.dataframe(cluster_summary)

# Visualizations
st.subheader("📈 Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.image("elbow_plot.png", caption="Elbow Plot: Optimal Number of Clusters")

with col2:
    st.image("cluster_heatmap.png", caption="Cluster Profile Heatmap")

# Segment labels (mock interpretation)
st.subheader("💡 Segment Descriptions")
segments = {
    0: "Loyal Lifers – Frequent buyers, high loyalty, strong referrals",
    1: "Health Hackers – Fitness-focused meals, highly engaged online",
    2: "Weekend Treaters – High spenders on weekends",
    3: "Deal Seekers – Promo-heavy usage, low retention",
    4: "One-and-Done – Tried once, didn’t return"
}

for cluster_id, desc in segments.items():
    st.markdown(f"**Cluster {cluster_id}:** {desc}")
