import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

st.set_page_config(page_title="Advanced Data Explorer", layout="wide")
st.title("Advanced Data Explorer Dashboard")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        all_columns = sorted(df.columns.tolist())

        st.sidebar.header("Filter Options")
        irrelevant_columns = st.sidebar.multiselect("Exclude irrelevant columns", options=all_columns)
        relevant_columns = list(set(all_columns)-set(irrelevant_columns))
        filtered_df = df[relevant_columns].copy()

        # User-defined datetime column
        datetime_input_cols = [col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0]
        ts_col = st.sidebar.selectbox("Select datetime column for time-series", datetime_input_cols)

        # Date Range Picker - placed below date column chooser
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            min_date = df[ts_col].min()
            max_date = df[ts_col].max()
            start_date, end_date = st.sidebar.date_input(f"Select date range for {ts_col}", [min_date, max_date])

        # Remove datetime from categorical filtering
        cat_cols = sorted([col for col in df.select_dtypes(include=['object', 'category']).columns if col not in irrelevant_columns and col != ts_col])
        for col in cat_cols:
            unique_vals = df[col].dropna().unique().tolist()
            selected_vals = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=unique_vals)
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        # Numeric Filters (excluding datetime)
        num_cols = sorted([col for col in df.select_dtypes(include='number').columns if col not in irrelevant_columns])
        for col in num_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

        # Date filtering
        if ts_col and start_date and end_date:
            filtered_df = filtered_df[(df[ts_col] >= pd.to_datetime(start_date)) & (df[ts_col] <= pd.to_datetime(end_date))]

        # Dataset Overview
        st.subheader("Dataset Overview")
        st.write("This section shows the first few rows of your filtered data.")
        st.dataframe(filtered_df.head())

        st.subheader("Summary Statistics")
        st.write("This section provides descriptive statistics for numeric and categorical columns.")
        st.write(filtered_df.describe(include='all'))

        # Column Profiler
        st.subheader("Column Profiler")
        st.write("Select a column to explore its distribution or frequency.")
        selected_col = st.selectbox("Select a column to profile", sorted(filtered_df.columns))
        if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
            st.plotly_chart(px.histogram(filtered_df, x=selected_col), use_container_width=True)
        else:
            st.write(filtered_df[selected_col].value_counts())

        # Custom Plotting
        st.subheader("Custom Plot")
        st.write("Create your own plot by choosing columns and chart type.")
        x_axis = st.selectbox("X-axis", sorted(filtered_df.columns))
        y_axis = st.selectbox("Y-axis", num_cols)
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"])

        if chart_type == "Bar":
            fig = px.bar(filtered_df, x=x_axis, y=y_axis)
        elif chart_type == "Line":
            fig = px.line(filtered_df, x=x_axis, y=y_axis)
        else:
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)

        # Pie Chart
        st.subheader("Pie Chart")
        st.write("Visualize distribution of a categorical column using a pie chart.")
        categorical_cols = sorted(filtered_df.select_dtypes(include='object').columns.tolist())
        if categorical_cols:
            pie_col = st.selectbox("Select categorical column", categorical_cols)
            pie_data = filtered_df[pie_col].value_counts()
            fig = px.pie(names=pie_data.index, values=pie_data.values, title=f"Distribution of {pie_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found for pie chart.")

        # Time-Series
        st.subheader("Time-Series Explorer")
        st.write("This section helps explore numeric trends over time if a valid date column is present.")

        try:
            if ts_col:
                # Ensure datetime column is properly cast
                filtered_df[ts_col] = pd.to_datetime(filtered_df[ts_col], errors='coerce')
                filtered_df = filtered_df.dropna(subset=[ts_col])  # Drop rows where date couldn't be parsed

                ts_metric = st.selectbox("Select numeric column", num_cols)
                freq = st.selectbox("Resample frequency", ["D", "W", "M"], index=1,
                                    format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
                
                ts_df = filtered_df[[ts_col, ts_metric]].dropna()
                ts_df = ts_df.set_index(ts_col).resample(freq).mean().reset_index()
                
                fig = px.line(ts_df, x=ts_col, y=ts_metric)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid datetime column selected for time-series analysis.")
        except Exception as e:
            st.warning(f"An error occurred with time-series analysis: {str(e)}")
        
        
        # Correlation Matrix / Heatmap
        st.subheader("Correlation Heatmap")
        st.write("Explore the correlation between numeric variables. Select columns to include (defaults to all):")

        corr_cols = st.multiselect("Select columns for correlation matrix", sorted(num_cols))

        # Use all numeric columns if none selected
        columns_to_corr = corr_cols if corr_cols else sorted(num_cols)

        if len(columns_to_corr) >= 2:
            try:
                corr_matrix = filtered_df[columns_to_corr].corr()

                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not create correlation matrix: {str(e)}")
        else:
            st.info("Please select at least two numeric columns to view the correlation heatmap.")


        # Anomaly Detection
        st.subheader("Anomaly Detection")
        st.write("Detect unusual values in numeric data using Z-score or Isolation Forest.")
        target_col = st.selectbox("Select column for anomaly detection", num_cols)
        method = st.selectbox("Detection method", ["Z-Score", "Isolation Forest"])
        plot_type = st.selectbox("Visualize anomalies using", ["Scatter", "Boxplot"])

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
            anomalies = filtered_df.loc[values.index[preds == -1]]

        st.write(f"Anomalies Detected: {len(anomalies)}")

        if not anomalies.empty:
            if plot_type == "Scatter":
                fig = px.scatter(filtered_df, x=filtered_df.index, y=target_col, title="Anomaly Detection")
                fig.add_scatter(x=anomalies.index, y=anomalies[target_col], mode='markers', marker=dict(color='red', size=10), name="Anomaly")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.box(filtered_df, y=target_col, points="all")
                st.plotly_chart(fig, use_container_width=True)

        # Clustering Explorer

        # Clustering Explorer (KMeans and KPrototypes)
        st.subheader("Clustering Explorer")
        st.write("Cluster the data using KMeans or KPrototypes. PCA is used to reduce dimensions for plotting.")
        # Common preprocessing
        cluster_features = st.multiselect("Select features for clustering", sorted(df.columns))
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        algorithm = st.selectbox("Clustering Algorithm", ["KMeans (Numeric Only)", "KPrototypes (Mixed Data)"])

        if len(cluster_features) >= 2:
            cluster_data = filtered_df[cluster_features].dropna()
            
            if algorithm == "KMeans (Numeric Only)":
                numeric_data = cluster_data.select_dtypes(include='number')
                scaled_data = StandardScaler().fit_transform(numeric_data)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(scaled_data)
                pca_input = scaled_data

            elif algorithm == "KPrototypes (Mixed Data)":
                # Encode categorical columns
                cat_cols = cluster_data.select_dtypes(include=['object', 'category']).columns
                enc_data = cluster_data.copy()
                cat_idx = []

                for i, col in enumerate(cluster_data.columns):
                    if col in cat_cols:
                        enc_data[col] = LabelEncoder().fit_transform(enc_data[col])
                        cat_idx.append(i)

                model = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
                labels = model.fit_predict(enc_data.values, categorical=cat_idx)
                pca_input = StandardScaler().fit_transform(enc_data.values)

            # PCA and Plotting (only once)
            pca = PCA(n_components=3)
            components = pca.fit_transform(pca_input)
            cluster_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
            cluster_df["Cluster"] = labels.astype(str)

            plot_dim = st.radio("Select plot dimensions", ["2D", "3D"])
            fig = (
                px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster")
                if plot_dim == "2D"
                else px.scatter_3d(cluster_df, x="PC1", y="PC2", z="PC3", color="Cluster")
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Please select at least two features for clustering.")
      
        # Download Filtered Data
        st.subheader("Download Filtered Data")
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload a CSV or Excel file to begin.")
