
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="AI-Powered Retail Analytics System", layout="wide")

st.sidebar.title("Retail Analytics System")

page = st.sidebar.radio(
    "Navigation",
    ["Home","Executive Summary","EDA Analysis","Customer Segmentation",
     "Sales Forecasting","Sales Prediction","Business Recommendations"]
)

uploaded_file = st.sidebar.file_uploader("Upload Retail Dataset", type=["csv"])

required_columns = [
    'Order Date','Category','Sub-Category','Region','Segment','Sales'
]

df = None
dataset_valid = False

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    missing = list(set(required_columns) - set(df.columns))
    dataset_valid = len(missing) == 0

def kpi_dashboard(df):
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    max_sale = df["Sales"].max()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹ {total_sales:,.2f}")
    c2.metric("Average Sale", f"₹ {avg_sales:,.2f}")
    c3.metric("Highest Sale", f"₹ {max_sale:,.2f}")
    c4.metric("Orders", len(df))

if page == "Home":
    st.title("AI-Powered Retail Analytics System")

    if df is None:
        st.info("Upload a dataset from the sidebar.")
    else:
        st.success("Dataset uploaded successfully.")
        kpi_dashboard(df)

        st.subheader("Dataset Overview")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        category_sales = df.groupby("Category")["Sales"].sum().reset_index()

        fig, ax = plt.subplots()
        sns.barplot(data=category_sales, x="Category", y="Sales", ax=ax)
        st.pyplot(fig)

        top_cat = category_sales.loc[category_sales["Sales"].idxmax(), "Category"]
        st.info(f"Insight: {top_cat} is the highest revenue generating category.")

elif page == "Executive Summary":
    st.title("Executive Summary")

    if df is None:
        st.warning("Upload dataset first.")
    else:
        total_sales = df["Sales"].sum()
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        top_segment = df.groupby("Segment")["Sales"].sum().idxmax()

        st.success(f"Total Revenue: ₹ {total_sales:,.2f}")
        st.write(f"Top Category: {top_category}")
        st.write(f"Top Region: {top_region}")
        st.write(f"Top Segment: {top_segment}")

        st.markdown("""
### Project Summary
- Performed Data Cleaning and EDA
- Built Multiple ML Models
- Customer Segmentation using K-Means
- Sales Forecasting using Prophet
- Interactive AI Dashboard
- Business Recommendation Engine
""")

elif page == "EDA Analysis":
    st.title("EDA Analysis")

    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.dataframe(df.head())

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing"))

        fig, ax = plt.subplots()
        sns.histplot(df["Sales"], kde=True, ax=ax)
        st.pyplot(fig)
        st.info("Sales distribution helps identify revenue concentration and outliers.")

        category_sales = df.groupby("Category")["Sales"].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=category_sales, x="Category", y="Sales", ax=ax)
        st.pyplot(fig)

        top_cat = category_sales.loc[category_sales["Sales"].idxmax(),"Category"]
        st.info(f"{top_cat} contributes the highest revenue.")

        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=region_sales, x="Region", y="Sales", ax=ax)
        st.pyplot(fig)

        top_region = region_sales.loc[region_sales["Sales"].idxmax(),"Region"]
        st.info(f"{top_region} is the best performing region.")

elif page == "Customer Segmentation":
    st.title("Customer Segmentation")

    if df is None:
        st.warning("Upload dataset first.")
    elif "Customer ID" not in df.columns or "Customer Name" not in df.columns:
        st.error("Customer columns not found.")
    else:
        customer_data = (
            df.groupby(["Customer ID","Customer Name"])["Sales"]
            .sum()
            .reset_index()
        )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(customer_data[["Sales"]])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        customer_data["Cluster"] = kmeans.fit_predict(scaled)

        summary = customer_data.groupby("Cluster")["Sales"].agg(
            ["count","mean","min","max"]
        )

        st.dataframe(summary)

        fig, ax = plt.subplots()
        ax.scatter(customer_data.index, customer_data["Sales"],
                   c=customer_data["Cluster"])
        st.pyplot(fig)

        best_cluster = summary["mean"].idxmax()
        st.success(
            f"Cluster {best_cluster} contains the highest-value customers."
        )

elif page == "Sales Forecasting":
    st.title("Sales Forecasting")

    if df is None:
        st.warning("Upload dataset first.")
    elif not PROPHET_AVAILABLE:
        st.error("Prophet library not installed.")
    else:
        data = df.copy()
        data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)

        sales = data.groupby("Order Date")["Sales"].sum().reset_index()
        sales.columns = ["ds","y"]

        model = Prophet()
        model.fit(sales)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        st.dataframe(
            forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(10)
        )

        fig = model.plot(forecast)
        st.pyplot(fig)

        future_avg = forecast["yhat"].tail(90).mean()
        st.info(f"Average predicted future sales: ₹ {future_avg:,.2f}")

elif page == "Sales Prediction":
    st.title("AI Sales Prediction")

    if df is None:
        st.warning("Upload dataset first.")
    else:
        try:
            model = joblib.load("best_sales_model.pkl")
            feature_columns = joblib.load("feature_columns.pkl")

            category = st.selectbox("Category", sorted(df["Category"].unique()))
            subcategory = st.selectbox("Sub-Category", sorted(df["Sub-Category"].unique()))
            region = st.selectbox("Region", sorted(df["Region"].unique()))
            segment = st.selectbox("Segment", sorted(df["Segment"].unique()))
            month = st.slider("Month",1,12,6)
            year = st.slider("Year",2020,2035,2026)

            if st.button("Predict Sales"):
                input_df = pd.DataFrame(
                    np.zeros((1,len(feature_columns))),
                    columns=feature_columns
                )

                if "Month" in input_df.columns:
                    input_df["Month"] = month
                if "Year" in input_df.columns:
                    input_df["Year"] = year

                for col in [
                    f"Category_{category}",
                    f"Sub-Category_{subcategory}",
                    f"Region_{region}",
                    f"Segment_{segment}"
                ]:
                    if col in input_df.columns:
                        input_df[col] = 1

                pred = model.predict(input_df)[0]

                st.metric("Predicted Sales", f"₹ {pred:,.2f}")
                st.success("Prediction generated successfully.")
        except Exception as e:
            st.error(str(e))

elif page == "Business Recommendations":
    st.title("Business Recommendations")

    if df is None:
        st.warning("Upload dataset first.")
    else:
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        top_segment = df.groupby("Segment")["Sales"].sum().idxmax()

        st.success(f"Top Category: {top_category}")
        st.info("Increase inventory and marketing budget for this category.")

        st.success(f"Top Region: {top_region}")
        st.info("Expand promotions and business operations in this region.")

        st.success(f"Top Segment: {top_segment}")
        st.info("Launch loyalty programs and retention campaigns.")
