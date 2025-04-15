import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Advanced Data Explorer", layout="wide")
st.title("Advanced Data Explorer Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sidebar filters
    st.sidebar.header("Filter Options")
    filterable_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    filtered_df = df.copy()
    for col in filterable_cols:
        selected_vals = st.sidebar.multiselect(f"Filter by {col}", df[col].dropna().unique(), default=list(df[col].dropna().unique()))
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    # Section: Dataset Overview
    st.subheader("Dataset Overview")
    st.dataframe(filtered_df.head())

    st.subheader("Summary Statistics")
    st.write(filtered_df.describe(include='all'))

    # Section: Column Profiler
    st.subheader("Column Profiler")
    selected_col = st.selectbox("Select a column to profile", filtered_df.columns)
    if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
        st.write("Histogram")
        st.plotly_chart(px.histogram(filtered_df, x=selected_col), use_container_width=True)
    else:
        st.write("Value Counts")
        st.write(filtered_df[selected_col].value_counts())

    # Section: Custom Plotting
    st.subheader("Custom Plot")
    x_axis = st.selectbox("X-axis", filtered_df.columns)
    y_axis = st.selectbox("Y-axis", filtered_df.select_dtypes(include='number').columns)
    chart_type = st.selectbox("Chart type", ["Line", "Scatter", "Bar"])

    if chart_type == "Line":
        fig = px.line(filtered_df, x=x_axis, y=y_axis)
    elif chart_type == "Scatter":
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
    else:
        fig = px.bar(filtered_df, x=x_axis, y=y_axis)
    st.plotly_chart(fig, use_container_width=True)

    # Section: Time-Series Detection
    st.subheader("Time-Series Explorer")
    datetime_cols = []
    for col in filtered_df.columns:
        try:
            filtered_df[col] = pd.to_datetime(filtered_df[col])
            datetime_cols.append(col)
        except:
            continue

    datetime_cols = list(set(datetime_cols))
    if datetime_cols:
        ts_col = st.selectbox("Select datetime column", datetime_cols)
        ts_metric = st.selectbox("Select numeric column to plot", filtered_df.select_dtypes(include='number').columns)
        freq = st.selectbox("Resample frequency", ["D", "W", "M"], index=1, format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
        ts_df = filtered_df[[ts_col, ts_metric]].dropna()
        ts_df = ts_df.groupby(pd.Grouper(key=ts_col, freq=freq)).mean().reset_index()
        fig = px.line(ts_df, x=ts_col, y=ts_metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No datetime column detected for time-series analysis.")

    # Section: Anomaly Detection
    st.subheader("Anomaly Detection")
    numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox("Select numeric column for anomaly detection", numeric_cols)
    method = st.selectbox("Detection method", ["Z-Score", "Isolation Forest"])

    if method == "Z-Score":
        threshold = st.slider("Z-score threshold", 0.0, 5.0, 3.0)
        mean_val = filtered_df[target_col].mean()
        std_val = filtered_df[target_col].std()
        filtered_df["z_score"] = (filtered_df[target_col] - mean_val) / std_val
        anomalies = filtered_df[np.abs(filtered_df["z_score"]) > threshold]
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        values = filtered_df[[target_col]].dropna()
        preds = model.fit_predict(values)
        anomalies = values[preds == -1]

    st.write(f"Anomalies Detected: {len(anomalies)}")
    if not anomalies.empty:
        fig = px.scatter(filtered_df, x=filtered_df.index, y=target_col)
        fig.add_scatter(x=anomalies.index, y=anomalies[target_col], mode='markers', marker=dict(color='red', size=10), name="Anomaly")
        st.plotly_chart(fig, use_container_width=True)

    # Section: Clustering Explorer
    st.subheader("Clustering Explorer")
    cluster_features = st.multiselect("Select numeric features for clustering", numeric_cols)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if len(cluster_features) >= 2:
        data = filtered_df[cluster_features].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(scaled_data)

        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled_data)
        cluster_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
        cluster_df["Cluster"] = labels.astype(str)

        plot_dim = st.radio("Select plot dimensions", ["2D", "3D"])
        if plot_dim == "2D":
            fig = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster")
        else:
            fig = px.scatter_3d(cluster_df, x="PC1", y="PC2", z="PC3", color="Cluster")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least two numeric features for clustering.")

    # Section: Download
    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

else:
    st.info("Upload a CSV file to begin.")
