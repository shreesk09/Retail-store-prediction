import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI-Powered Retail Analytics System",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Retail Analytics System")

page = st.sidebar.radio(
    "Navigation",
    [  
    "Home",
    "Executive Summary",
    "EDA Analysis",
    "Customer Segmentation",
    "Sales Forecasting",
    "Sales Prediction",
    "Business Recommendations"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Dataset",
    type=["csv"]
)

# =====================================================
# REQUIRED COLUMNS
# =====================================================

required_columns = [
    'Order Date',
    'Category',
    'Sub-Category',
    'Region',
    'Segment',
    'Sales'
]

# =====================================================
# LOAD DATA
# =====================================================

df = None
dataset_valid = False

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        encoding='latin1'
    )

    missing_columns = list(
        set(required_columns) - set(df.columns)
    )

    if len(missing_columns) == 0:
        dataset_valid = True

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("AI-Powered Retail Analytics System")

    if uploaded_file is None:

        st.info("Upload a retail dataset from sidebar.")

    else:

        st.success("Dataset Uploaded Successfully")

        st.subheader("Business KPI Dashboard")

        total_sales = df["Sales"].sum()
        avg_sales = df["Sales"].mean()
        max_sale = df["Sales"].max()
        total_orders = len(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("💰 Total Revenue", f"₹ {total_sales:,.2f}")
        col2.metric("📊 Average Sale", f"₹ {avg_sales:,.2f}")
        col3.metric("🏆 Highest Sale", f"₹ {max_sale:,.2f}")
        col4.metric("📦 Orders", total_orders)

        top_category = (
            df.groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        top_region = (
            df.groupby("Region")["Sales"]
            .sum()
            .idxmax()
        )

        col5, col6 = st.columns(2)

        col5.metric("⭐ Top Category", top_category)
        col6.metric("🌍 Top Region", top_region)

        st.markdown("---")

        st.subheader("Revenue by Category")

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots()

        sns.barplot(
            data=category_sales,
            x="Category",
            y="Sales",
            ax=ax
        )

        st.pyplot(fig)

        st.info(
            f"""
            Insight:

            {top_category} generates the highest revenue.

            Recommendation:
            Increase inventory and marketing budget for this category.
            """
        )

# =====================================================
# KPI DASHBOARD
# =====================================================

if uploaded_file is not None:

    st.subheader("Business KPI Dashboard")

    total_sales = df["Sales"].sum()

    avg_sales = df["Sales"].mean()

    max_sale = df["Sales"].max()

    total_orders = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"₹ {total_sales:,.2f}"
    )

    col2.metric(
        "Average Sale",
        f"₹ {avg_sales:,.2f}"
    )

    col3.metric(
        "Highest Sale",
        f"₹ {max_sale:,.2f}"
    )

    col4.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    top_category = (
    df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

    top_region = (
    df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

    col5, col6 = st.columns(2)

    col5.metric(
    "Top Category",
    top_category
)

    col6.metric(
    "Top Region",
    top_region
)

    st.subheader("Revenue by Category")

    category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(
    data=category_sales,
    x="Category",
    y="Sales",
    ax=ax
)

    st.pyplot(fig)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    monthly_sales = (
        df.groupby(
            df["Order Date"].dt.month
        )["Sales"]
        .sum()
    )

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots()

    monthly_sales.plot(
        marker="o",
        ax=ax
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

elif page == "Executive Summary":

    st.title("Executive Summary Report")

    if uploaded_file is None:

        st.warning("Upload Dataset First")

    else:

        total_sales = df["Sales"].sum()

        top_category = (
            df.groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        top_region = (
            df.groupby("Region")["Sales"]
            .sum()
            .idxmax()
        )

        top_segment = (
            df.groupby("Segment")["Sales"]
            .sum()
            .idxmax()
        )

        st.success(
            f"Total Revenue Generated: ₹ {total_sales:,.2f}"
        )

        st.info(
            f"Best Performing Category: {top_category}"
        )

        st.info(
            f"Best Performing Region: {top_region}"
        )

        st.info(
            f"Most Valuable Customer Segment: {top_segment}"
        )

        st.subheader("Business Recommendations")

        st.write(
            """
            • Increase inventory for top-performing categories.

            • Focus marketing campaigns in the highest revenue region.

            • Create loyalty programs for high-value customer segments.

            • Use forecasting results for inventory planning.

            • Use prediction module for future sales estimation.
            """
        )
# =====================================================
# EDA PAGE
# =====================================================

elif page == "EDA Analysis":

    st.title("Exploratory Data Analysis")

    if uploaded_file is None:

        st.warning("Please upload dataset first.")

    else:

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.subheader("Dataset Validation")

        missing_columns = list(
            set(required_columns) - set(df.columns)
        )

        if len(missing_columns) == 0:

            st.success(
                "Dataset is compatible with all modules."
            )

        else:

            st.error(
                f"Missing Columns: {missing_columns}"
            )

        st.subheader("Missing Values")

        st.dataframe(
            pd.DataFrame(
                df.isnull().sum(),
                columns=['Missing Values']
            )
        )

        # Sales Distribution

        if "Sales" in df.columns:

            st.subheader("Sales Distribution")

            fig, ax = plt.subplots()

            sns.histplot(
                df["Sales"],
                kde=True,
                ax=ax
            )

            st.pyplot(fig)

        # Category Sales

        if (
            "Category" in df.columns
            and "Sales" in df.columns
        ):

            st.subheader("Category Wise Sales")

            category_sales = (
                df.groupby("Category")["Sales"]
                .sum()
                .reset_index()
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            sns.barplot(
                data=category_sales,
                x="Category",
                y="Sales",
                ax=ax
            )

            st.pyplot(fig)
            top_category = category_sales.loc[
            category_sales["Sales"].idxmax(),
            "Category"]

            st.info(
            f"""
            Insight:

            {top_category} is the highest revenue generating category.

            Recommendation:
            Allocate more resources to this category.
            """
            )
        # Region Sales

        if (
            "Region" in df.columns
            and "Sales" in df.columns
        ):
            st.subheader("Region Wise Sales")

            region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
            )

            fig, ax = plt.subplots(
            figsize=(8, 4)
            )

        sns.barplot(
        data=region_sales,
        x="Region",
        y="Sales",
        ax=ax
        )

        st.pyplot(fig) 
        top_region = region_sales.loc[
        region_sales["Sales"].idxmax(),
        "Region"
        ]

        st.info(
        f"""
        Insight:

        {top_region} contributes the highest sales.

        Recommendation:
        Expand operations and promotions in this region.
        """
        ) 

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    if uploaded_file is None:

        st.warning("Please upload dataset first.")

    elif not dataset_valid:

        st.error(
            "Dataset not compatible."
        )

    else:

        if (
            "Customer ID" in df.columns
            and "Customer Name" in df.columns
        ):

            customer_data = (
                df.groupby(
                    ['Customer ID',
                     'Customer Name']
                )['Sales']
                .sum()
                .reset_index()
            )

            scaler = StandardScaler()

            scaled_data = scaler.fit_transform(
                customer_data[['Sales']]
            )

            kmeans = KMeans(
                n_clusters=3,
                random_state=42
            )

            customer_data['Cluster'] = (
                kmeans.fit_predict(
                    scaled_data
                )
            )

            st.subheader(
                "Customer Segments"
            )

            st.dataframe(
                customer_data.head()
            )

            summary = (
                customer_data
                .groupby("Cluster")["Sales"]
                .agg([
                    'count',
                    'mean',
                    'min',
                    'max'
                ])
            )

            st.subheader(
                "Cluster Summary"
            )

            st.dataframe(summary)

            highest_cluster = summary["mean"].idxmax()

            st.success(
            f"""
            Cluster {highest_cluster}
            contains the highest-value customers.

            Recommendation:
            Launch loyalty and retention programs
            for these customers.
            """
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            scatter = ax.scatter(
                customer_data.index,
                customer_data['Sales'],
                c=customer_data['Cluster']
            )

            ax.set_title(
                "Customer Segmentation"
            )

            ax.set_xlabel(
                "Customers"
            )

            ax.set_ylabel(
                "Sales"
            )

            st.pyplot(fig)

        else:

            st.error(
                "Customer columns not found."
            )

# =====================================================
# SALES FORECASTING
# =====================================================

elif page == "Sales Forecasting":

    st.title("Sales Forecasting")

    if uploaded_file is None:

        st.warning("Please upload dataset first.")

    elif not dataset_valid:

        st.error(
            "Dataset not compatible."
        )

    else:

        st.subheader(
            "Future Sales Forecast"
        )

        df['Order Date'] = pd.to_datetime(
            df['Order Date'],
            dayfirst=True
        )

        sales_data = (
            df.groupby(
                'Order Date'
            )['Sales']
            .sum()
            .reset_index()
        )

        sales_data.columns = [
            'ds',
            'y'
        ]

        model = Prophet()

        model.fit(
            sales_data
        )

        future = (
            model.make_future_dataframe(
                periods=90
            )
        )

        forecast = model.predict(
            future
        )

        st.subheader(
            "Forecast Results"
        )

        st.dataframe(
            forecast[
                [
                    'ds',
                    'yhat',
                    'yhat_lower',
                    'yhat_upper'
                ]
            ].tail(10)
        )

        fig = model.plot(
            forecast
        )

        st.pyplot(fig)
        future_avg = (
    forecast["yhat"]
    .tail(90)
    .mean()
)

    st.info(
    f"""
    Forecast Insight:

    Average predicted future sales:
    ₹ {future_avg:,.2f}

    Recommendation:
    Maintain inventory levels according
    to expected future demand.
    """
)

    st.subheader(
            "Forecast Components"
        )

    fig2 = (
            model.plot_components(
                forecast
            )
        )

    st.pyplot(fig2)

# =====================================================
# SALES PREDICTION
# =====================================================

elif page == "Sales Prediction":

    st.title("AI Sales Prediction")

    try:

        model = joblib.load("best_sales_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")

        st.success("Model Loaded Successfully")

        category = st.selectbox(
            "Category",
            sorted(df["Category"].unique())
        )

        subcategory = st.selectbox(
            "Sub-Category",
            sorted(df["Sub-Category"].unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

        segment = st.selectbox(
            "Segment",
            sorted(df["Segment"].unique())
        )

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

        year = st.slider(
            "Year",
            2020,
            2030,
            2026
        )

        if st.button("Predict Sales"):

            input_data = pd.DataFrame(
                np.zeros((1, len(feature_columns))),
                columns=feature_columns
            )

            if "Month" in input_data.columns:
                input_data["Month"] = month

            if "Year" in input_data.columns:
                input_data["Year"] = year

            category_col = f"Category_{category}"

            if category_col in input_data.columns:
                input_data[category_col] = 1

            sub_col = f"Sub-Category_{subcategory}"

            if sub_col in input_data.columns:
                input_data[sub_col] = 1

            region_col = f"Region_{region}"

            if region_col in input_data.columns:
                input_data[region_col] = 1

            segment_col = f"Segment_{segment}"

            if segment_col in input_data.columns:
                input_data[segment_col] = 1

            prediction = model.predict(input_data)[0]

            st.metric(
                "Predicted Sales",
                f"₹ {prediction:,.2f}"
            )
            st.success(
    f"""
    Prediction Interpretation

    Category: {category}

    Sub-Category: {subcategory}

    Region: {region}

    Segment: {segment}

    Month: {month}

    Year: {year}

    Expected Sales:
    ₹ {prediction:,.2f}
    """
)

    except Exception as e:

        st.error(str(e))

# =====================================================
# BUSINESS RECOMMENDATIONS
# =====================================================

elif page == "Business Recommendations":

    st.title("AI Business Recommendation Engine")

    if uploaded_file is None:

        st.warning("Upload Dataset First")

    else:

        # Top Category

        top_category = (
            df.groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        # Top Region

        top_region = (
            df.groupby("Region")["Sales"]
            .sum()
            .idxmax()
        )

        # Top Segment

        top_segment = (
            df.groupby("Segment")["Sales"]
            .sum()
            .idxmax()
        )

        total_sales = df["Sales"].sum()

        st.subheader("Generated Recommendations")

        st.success(
            f"Top Performing Category: {top_category}"
        )

        st.info(
            f"Recommendation: Increase inventory and marketing budget for {top_category}"
        )

        st.success(
            f"Best Performing Region: {top_region}"
        )

        st.info(
            f"Recommendation: Focus promotions and expansion efforts in {top_region}"
        )

        st.success(
            f"Highest Revenue Customer Segment: {top_segment}"
        )

        st.info(
            f"Recommendation: Launch loyalty programs for {top_segment} customers"
        )

        st.metric(
            "Total Revenue",
            f"₹ {total_sales:,.2f}"
        )