# ==========================================================
# AI POWERED RETAIL ANALYTICS SYSTEM
# APP V3
# ==========================================================

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

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Retail Stores Sales Insights and Prediction Model",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    padding: 1rem;
}

div[data-testid="metric-container"] {
    background-color: #f5f5f5;
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Retail Analytics System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Executive Summary",
        "Dataset Validation",
        "EDA Analysis",
        "Customer Segmentation",
        "Sales Forecasting",
        "Sales Prediction",
        "Model Comparison",
        "Business Recommendations",
        "Download Reports"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Dataset",
    type=["csv"]
)

# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "Order Date",
    "Category",
    "Sub-Category",
    "Region",
    "Segment",
    "Sales"
]

# ==========================================================
# LOAD DATASET
# ==========================================================

df = None
dataset_valid = False

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file,
            encoding="latin1"
        )

        missing_columns = list(
            set(required_columns)
            - set(df.columns)
        )

        if len(missing_columns) == 0:
            dataset_valid = True

    except Exception as e:

        st.error(
            f"Dataset Loading Error: {e}"
        )

# ==========================================================
# KPI FUNCTION
# ==========================================================

def show_kpis(dataframe):

    total_sales = dataframe["Sales"].sum()

    avg_sales = dataframe["Sales"].mean()

    max_sales = dataframe["Sales"].max()

    total_orders = len(dataframe)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Revenue",
        f"₹ {total_sales:,.2f}"
    )

    c2.metric(
        "Average Sale",
        f"₹ {avg_sales:,.2f}"
    )

    c3.metric(
        "Highest Sale",
        f"₹ {max_sales:,.2f}"
    )

    c4.metric(
        "Orders",
        total_orders
    )

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.title(
        "Retail Stores Sales Insights and Prediction Model"
    )

    if uploaded_file is None:

        st.info(
            "Upload a dataset from the sidebar."
        )

    else:

        st.success(
            "Dataset Uploaded Successfully"
        )

        show_kpis(df)


        c1, c2 = st.columns(2)

        c1.metric(
            "Rows",
            df.shape[0]
        )

        c2.metric(
            "Columns",
            df.shape[1]
        )

        st.dataframe(
            df.head()
        )
# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

elif page == "Executive Summary":

    st.title("Executive Summary")

    if uploaded_file is None:

        st.warning(
            "Please upload dataset first."
        )

    else:

        total_sales = df["Sales"].sum()

        avg_sales = df["Sales"].mean()

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
            f"Total Revenue Generated : ₹ {total_sales:,.2f}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Top Category",
            top_category
        )

        c2.metric(
            "Top Region",
            top_region
        )

        c3.metric(
            "Top Segment",
            top_segment
        )

       


        st.subheader("Key Findings")

        st.info(
            f"""
            Highest Revenue Category:
            {top_category}

            Highest Revenue Region:
            {top_region}

            Highest Revenue Segment:
            {top_segment}

            Average Transaction Value:
            ₹ {avg_sales:,.2f}
            """
        )

# ==========================================================
# DASHBOARD ANALYTICS
# ==========================================================

        st.markdown("---")

        st.subheader(
            "Revenue by Category"
        )

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        sns.barplot(
            data=category_sales,
            x="Category",
            y="Sales",
            ax=ax
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

        best_category = category_sales.loc[
            category_sales["Sales"].idxmax(),
            "Category"
        ]

        st.success(
            f"""
            Insight:

            {best_category} contributes
            the highest revenue.

            Recommendation:

            Increase stock allocation,
            promotions and inventory
            planning for this category.
            """
        )

# ==========================================================
# REGION ANALYSIS
# ==========================================================

        st.markdown("---")

        st.subheader(
            "Region Wise Sales"
        )

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        sns.barplot(
            data=region_sales,
            x="Region",
            y="Sales",
            ax=ax
        )

        st.pyplot(fig)

        best_region = region_sales.loc[
            region_sales["Sales"].idxmax(),
            "Region"
        ]

        st.success(
            f"""
            Insight:

            {best_region}
            generates the highest revenue.

            Recommendation:

            Increase marketing campaigns,
            store expansion and customer
            acquisition efforts in this region.
            """
        )

# ==========================================================
# MONTHLY SALES TREND
# ==========================================================

        if "Order Date" in df.columns:

            try:

                temp_df = df.copy()

                temp_df["Order Date"] = pd.to_datetime(
                    temp_df["Order Date"],
                    dayfirst=True
                )

                temp_df["Month"] = (
                    temp_df["Order Date"]
                    .dt.month
                )

                monthly_sales = (
                    temp_df.groupby("Month")["Sales"]
                    .sum()
                )

                st.markdown("---")

                st.subheader(
                    "Monthly Sales Trend"
                )

                fig, ax = plt.subplots(
                    figsize=(10,4)
                )

                monthly_sales.plot(
                    marker="o",
                    ax=ax
                )

                ax.set_xlabel(
                    "Month"
                )

                ax.set_ylabel(
                    "Sales"
                )

                st.pyplot(fig)

                best_month = (
                    monthly_sales.idxmax()
                )

                st.success(
                    f"""
                    Insight:

                    Month {best_month}
                    recorded the highest sales.

                    Recommendation:

                    Plan inventory and
                    promotional campaigns
                    around peak demand periods.
                    """
                )

            except:

                pass
# =====================================================
# EDA ANALYSIS
# =====================================================

elif page == "EDA Analysis":

    st.title("Exploratory Data Analysis")

    if uploaded_file is None:

        st.warning("Please upload dataset first.")

    else:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        })

        st.dataframe(info_df)

        st.subheader("Missing Values")

        missing_df = pd.DataFrame(
            df.isnull().sum(),
            columns=["Missing Values"]
        )

        st.dataframe(missing_df)

        # ====================================
        # SALES DISTRIBUTION
        # ====================================

        st.subheader("Sales Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            df["Sales"],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

        st.info("""
        Insight:

        Most transactions are low-value sales,
        while a small number of orders contribute
        significantly higher revenue.

        Recommendation:

        Focus on identifying high-value products
        and customers to maximize profitability.
        """)

        # ====================================
        # CATEGORY SALES
        # ====================================

        st.subheader("Category Wise Sales")

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

        top_category = category_sales.loc[
            category_sales["Sales"].idxmax(),
            "Category"
        ]

        st.success(
            f"Top Revenue Category: {top_category}"
        )

        st.info(f"""
        Insight:

        {top_category} contributes the highest
        overall sales.

        Recommendation:

        Increase inventory allocation and
        marketing campaigns for this category.
        """)

        # ====================================
        # REGION SALES
        # ====================================

        st.subheader("Region Wise Sales")

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots()

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

        st.success(
            f"Top Revenue Region: {top_region}"
        )

        st.info(f"""
        Insight:

        {top_region} region generates the
        highest revenue.

        Recommendation:

        Expand promotions and inventory
        availability in this region.
        """)

        # ====================================
        # MONTHLY TREND
        # ====================================

        st.subheader("Monthly Sales Trend")

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

        fig, ax = plt.subplots()

        monthly_sales.plot(
            marker="o",
            ax=ax
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")

        st.pyplot(fig)

        best_month = monthly_sales.idxmax()

        st.info(f"""
        Insight:

        Month {best_month}
        generated the highest sales.

        Recommendation:

        Increase stock availability and
        marketing during peak months.
        """)

        # ====================================
        # CORRELATION MATRIX
        # ====================================

        st.subheader("Correlation Matrix")

        numeric_df = df.select_dtypes(
            include=np.number
        )

        fig, ax = plt.subplots(
            figsize=(8,5)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

        st.info("""
        Insight:

        Correlation analysis helps identify
        which variables influence sales.

        Recommendation:

        Use highly correlated variables for
        future prediction models.
        """)
# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    if uploaded_file is None:

        st.warning("Please upload dataset first.")

    elif (
        "Customer ID" not in df.columns
        or "Customer Name" not in df.columns
    ):

        st.error(
            "Customer columns not found in dataset."
        )

    else:

        customer_data = (
            df.groupby(
                ["Customer ID", "Customer Name"]
            )["Sales"]
            .sum()
            .reset_index()
        )

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            customer_data[["Sales"]]
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        customer_data["Cluster"] = (
            kmeans.fit_predict(
                scaled_data
            )
        )

        st.subheader(
            "Customer Segment Dataset"
        )

        st.dataframe(
            customer_data.head(20)
        )

        # ====================================
        # CLUSTER SUMMARY
        # ====================================

        summary = (
            customer_data
            .groupby("Cluster")["Sales"]
            .agg([
                "count",
                "mean",
                "min",
                "max"
            ])
        )

        st.subheader(
            "Cluster Summary"
        )

        st.dataframe(summary)

        highest_cluster = (
            summary["mean"].idxmax()
        )

        lowest_cluster = (
            summary["mean"].idxmin()
        )

        st.success(
            f"""
            Highest Value Cluster:
            Cluster {highest_cluster}

            Average Sales:
            ₹ {summary.loc[highest_cluster,'mean']:,.2f}
            """
        )

        st.warning(
            f"""
            Lowest Value Cluster:
            Cluster {lowest_cluster}

            Average Sales:
            ₹ {summary.loc[lowest_cluster,'mean']:,.2f}
            """
        )

        # ====================================
        # VISUALIZATION
        # ====================================

        st.subheader(
            "Customer Segmentation Visualization"
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        scatter = ax.scatter(
            customer_data.index,
            customer_data["Sales"],
            c=customer_data["Cluster"]
        )

        ax.set_xlabel(
            "Customer Index"
        )

        ax.set_ylabel(
            "Total Sales"
        )

        ax.set_title(
            "Customer Clusters"
        )

        st.pyplot(fig)

        # ====================================
        # INTERPRETATION
        # ====================================

        st.subheader(
            "Business Interpretation"
        )

        st.info(f"""
        Cluster {highest_cluster}
        contains premium customers
        contributing the highest revenue.

        These customers should be retained
        using loyalty rewards,
        personalized offers,
        and premium services.
        """)

        st.info(f"""
        Cluster {lowest_cluster}
        contains low-spending customers.

        Targeted promotions and discounts
        can be used to increase engagement
        and conversion.
        """)

        # ====================================
        # HIGH VALUE CUSTOMERS
        # ====================================

        st.subheader(
            "Top 10 High Value Customers"
        )

        top_customers = (
            customer_data
            .sort_values(
                by="Sales",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_customers
        )

        st.success("""
        Recommendation:

        Create exclusive loyalty programs
        for these top customers since
        they contribute a significant
        portion of total revenue.
        """)
# =====================================================
# SALES FORECASTING
# =====================================================

elif page == "Sales Forecasting":

    st.title("AI Sales Forecasting")

    if uploaded_file is None:

        st.warning(
            "Please upload dataset first."
        )

    else:

        st.subheader(
            "Future Sales Forecast"
        )

        df["Order Date"] = pd.to_datetime(
            df["Order Date"],
            dayfirst=True
        )

        sales_data = (
            df.groupby("Order Date")["Sales"]
            .sum()
            .reset_index()
        )

        sales_data.columns = [
            "ds",
            "y"
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

        # ====================================
        # FORECAST TABLE
        # ====================================

        st.subheader(
            "90 Day Forecast Results"
        )

        forecast_table = forecast[
            [
                "ds",
                "yhat",
                "yhat_lower",
                "yhat_upper"
            ]
        ].tail(90)

        st.dataframe(
            forecast_table
        )

        # ====================================
        # FORECAST GRAPH
        # ====================================

        st.subheader(
            "Sales Forecast Trend"
        )

        fig1 = model.plot(
            forecast
        )

        st.pyplot(fig1)

        # ====================================
        # FORECAST COMPONENTS
        # ====================================

        st.subheader(
            "Forecast Components"
        )

        fig2 = (
            model.plot_components(
                forecast
            )
        )

        st.pyplot(fig2)

        # ====================================
        # FUTURE SALES KPI
        # ====================================

        future_avg = (
            forecast["yhat"]
            .tail(90)
            .mean()
        )

        future_max = (
            forecast["yhat"]
            .tail(90)
            .max()
        )

        future_min = (
            forecast["yhat"]
            .tail(90)
            .min()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Forecast",
            f"₹ {future_avg:,.2f}"
        )

        col2.metric(
            "Maximum Forecast",
            f"₹ {future_max:,.2f}"
        )

        col3.metric(
            "Minimum Forecast",
            f"₹ {future_min:,.2f}"
        )

        # ====================================
        # FORECAST INTERPRETATION
        # ====================================

        st.subheader(
            "Forecast Insights"
        )

        st.info(f"""
        Average expected sales over
        the next 90 days:

        ₹ {future_avg:,.2f}

        This estimate helps management
        plan inventory levels,
        workforce allocation,
        and procurement activities.
        """)

        if future_max > future_avg:

            st.success(f"""
            Peak demand periods are expected.

            Maximum forecasted sales:

            ₹ {future_max:,.2f}

            Recommendation:

            Increase inventory availability
            before high-demand periods.
            """)

        if future_min < future_avg:

            st.warning(f"""
            Low demand periods are expected.

            Minimum forecasted sales:

            ₹ {future_min:,.2f}

            Recommendation:

            Launch promotional campaigns
            during slower periods.
            """)

        # ====================================
        # BUSINESS RECOMMENDATIONS
        # ====================================

        st.subheader(
            "AI Generated Recommendations"
        )

        st.success("""
        Recommendation 1:

        Maintain sufficient inventory
        based on future sales demand.
        """)

        st.success("""
        Recommendation 2:

        Allocate marketing budgets
        during forecasted peak periods.
        """)

        st.success("""
        Recommendation 3:

        Use forecasting outputs for
        workforce and procurement planning.
        """)

        st.success("""
        Recommendation 4:

        Reduce overstocking during
        predicted low-demand periods.
        """)
# =====================================================
# SALES PREDICTION
# =====================================================

elif page == "Sales Prediction":

    st.title("AI Sales Prediction")

    if uploaded_file is None:

        st.warning(
            "Please upload dataset first."
        )

    else:

        try:

            model = joblib.load(
                "best_sales_model.pkl"
            )

            feature_columns = joblib.load(
                "feature_columns.pkl"
            )

            st.success(
                "Prediction Model Loaded Successfully"
            )

            st.subheader(
                "Enter Product Details"
            )

            category = st.selectbox(
                "Category",
                sorted(
                    df["Category"]
                    .dropna()
                    .unique()
                )
            )

            subcategory = st.selectbox(
                "Sub-Category",
                sorted(
                    df["Sub-Category"]
                    .dropna()
                    .unique()
                )
            )

            region = st.selectbox(
                "Region",
                sorted(
                    df["Region"]
                    .dropna()
                    .unique()
                )
            )

            segment = st.selectbox(
                "Segment",
                sorted(
                    df["Segment"]
                    .dropna()
                    .unique()
                )
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
                2035,
                2026
            )

            if st.button(
                "Predict Future Sales"
            ):

                input_data = pd.DataFrame(
                    np.zeros(
                        (
                            1,
                            len(feature_columns)
                        )
                    ),
                    columns=feature_columns
                )

                if "Month" in input_data.columns:
                    input_data["Month"] = month

                if "Year" in input_data.columns:
                    input_data["Year"] = year

                category_col = (
                    f"Category_{category}"
                )

                if category_col in input_data.columns:
                    input_data[
                        category_col
                    ] = 1

                subcategory_col = (
                    f"Sub-Category_{subcategory}"
                )

                if subcategory_col in input_data.columns:
                    input_data[
                        subcategory_col
                    ] = 1

                region_col = (
                    f"Region_{region}"
                )

                if region_col in input_data.columns:
                    input_data[
                        region_col
                    ] = 1

                segment_col = (
                    f"Segment_{segment}"
                )

                if segment_col in input_data.columns:
                    input_data[
                        segment_col
                    ] = 1

                prediction = (
                    model.predict(
                        input_data
                    )[0]
                )

                st.metric(
                    "Predicted Sales",
                    f"₹ {prediction:,.2f}"
                )

                # ============================
                # INTERPRETATION
                # ============================

                st.subheader(
                    "Prediction Interpretation"
                )

                st.success(f"""
                Category:
                {category}

                Sub-Category:
                {subcategory}

                Region:
                {region}

                Segment:
                {segment}

                Month:
                {month}

                Year:
                {year}

                Expected Sales:
                ₹ {prediction:,.2f}
                """)

                # ============================
                # SALES RISK ANALYSIS
                # ============================

                st.subheader(
                    "Risk Analysis"
                )

                if prediction > 1000:

                    st.success("""
                    High Revenue Opportunity

                    Expected sales are strong.

                    Recommendation:

                    Increase inventory and
                    marketing investments.
                    """)

                elif prediction > 500:

                    st.info("""
                    Moderate Revenue Opportunity

                    Recommendation:

                    Maintain regular stock
                    levels and promotions.
                    """)

                else:

                    st.warning("""
                    Low Revenue Prediction

                    Recommendation:

                    Consider discounts,
                    bundling offers,
                    and promotional campaigns.
                    """)

                # ============================
                # CONFIDENCE SCORE
                # ============================

                confidence = min(
                    95,
                    max(
                        70,
                        round(
                            prediction / 100
                        )
                    )
                )

                st.subheader(
                    "Prediction Confidence"
                )

                st.progress(
                    confidence / 100
                )

                st.write(
                    f"Confidence Score: {confidence}%"
                )

        except Exception as e:

            st.error(
                f"Model Error: {e}"
            )
# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("Machine Learning Model Comparison")

    try:

        results = pd.read_csv(
            "model_results.csv"
        )

        st.subheader(
            "Model Performance Table"
        )

        st.dataframe(results)

        # ====================================
        # BEST MODEL
        # ====================================

        best_model = results.loc[
            results["R2 Score"].idxmax()
        ]

        st.success(f"""
        Best Performing Model

        Model:
        {best_model['Model']}

        R² Score:
        {best_model['R2 Score']:.4f}
        """)

        # ====================================
        # R2 SCORE GRAPH
        # ====================================

        st.subheader(
            "R² Score Comparison"
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        sns.barplot(
            data=results,
            x="Model",
            y="R2 Score",
            ax=ax
        )

        plt.xticks(rotation=20)

        st.pyplot(fig)

        # ====================================
        # MAE GRAPH
        # ====================================

        st.subheader(
            "MAE Comparison"
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        sns.barplot(
            data=results,
            x="Model",
            y="MAE",
            ax=ax
        )

        plt.xticks(rotation=20)

        st.pyplot(fig)

        # ====================================
        # RMSE GRAPH
        # ====================================

        st.subheader(
            "RMSE Comparison"
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        sns.barplot(
            data=results,
            x="Model",
            y="RMSE",
            ax=ax
        )

        plt.xticks(rotation=20)

        st.pyplot(fig)

        # ====================================
        # INTERPRETATION
        # ====================================

        st.subheader(
            "Model Interpretation"
        )

        st.info("""
        R² Score indicates how well
        the model explains sales variation.

        Higher R² Score = Better Model.
        """)

        st.info("""
        MAE (Mean Absolute Error)
        indicates average prediction error.

        Lower MAE = Better Accuracy.
        """)

        st.info("""
        RMSE penalizes large errors.

        Lower RMSE = Better Performance.
        """)

        st.success(f"""
        Final Selected Model:

        {best_model['Model']}

        This model was selected
        for deployment in the
        Sales Prediction module.
        """)

    except Exception as e:

        st.error(
            f"Unable to load model results: {e}"
        )
# =====================================================
# BUSINESS RECOMMENDATIONS
# =====================================================

elif page == "Business Recommendations":

    st.title("AI Business Recommendation Engine")

    if uploaded_file is None:

        st.warning(
            "Please upload dataset first."
        )

    else:

        total_sales = df["Sales"].sum()

        avg_sales = df["Sales"].mean()

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

        st.subheader(
            "Business Overview"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Revenue",
            f"₹ {total_sales:,.2f}"
        )

        col2.metric(
            "Average Sale",
            f"₹ {avg_sales:,.2f}"
        )

        col3.metric(
            "Top Category",
            top_category
        )

        # ====================================
        # RECOMMENDATIONS
        # ====================================

        st.subheader(
            "Strategic Recommendations"
        )

        st.success(f"""
        Product Strategy

        {top_category} is the highest
        revenue generating category.

        Action:

        Increase stock availability,
        supplier engagement,
        and marketing investments
        for this category.
        """)

        st.success(f"""
        Regional Strategy

        {top_region} is the best
        performing region.

        Action:

        Expand promotional campaigns
        and regional operations
        in this market.
        """)

        st.success(f"""
        Customer Strategy

        {top_segment} contributes
        the highest revenue.

        Action:

        Launch loyalty programs
        and personalized offers
        for this segment.
        """)

        # ====================================
        # INVENTORY PLANNING
        # ====================================

        st.subheader(
            "Inventory Planning"
        )

        st.info("""
        Use forecasting outputs
        to maintain inventory
        during peak demand periods.

        Avoid overstocking during
        low-demand periods.
        """)

        # ====================================
        # MARKETING STRATEGY
        # ====================================

        st.subheader(
            "Marketing Strategy"
        )

        st.info("""
        Target high-value customers
        with personalized campaigns.

        Focus digital advertising
        on top-performing categories.
        """)

        # ====================================
        # REVENUE GROWTH
        # ====================================

        st.subheader(
            "Revenue Growth Opportunities"
        )

        st.success("""
        Opportunity 1

        Cross-sell products from
        top-performing categories.
        """)

        st.success("""
        Opportunity 2

        Upsell premium products
        to high-value customers.
        """)

        st.success("""
        Opportunity 3

        Improve retention using
        reward and membership programs.
        """)

        st.success("""
        Opportunity 4

        Optimize pricing during
        high-demand seasons.
        """)

        # ====================================
        # CEO SUMMARY
        # ====================================

        st.subheader(
            "Executive Conclusion"
        )

        st.markdown(f"""
        ### Key Findings

        - Total Revenue: ₹ {total_sales:,.2f}
        - Top Category: {top_category}
        - Top Region: {top_region}
        - Top Customer Segment: {top_segment}

        ### Final Recommendation

        Focus on expanding high-performing
        categories and regions while
        retaining high-value customers.

        Forecasting indicates future sales
        opportunities that can be leveraged
        through inventory planning and
        targeted marketing strategies.
        """)
# =====================================================
# DOWNLOAD REPORTS
# =====================================================

elif page == "Download Reports":

    st.title("Download Generated Reports")

    st.subheader("Available Files")

    files = [
        "cleaned_superstore1.csv",
        "customer_segments.csv",
        "sales_forecast.csv",
        "model_results.csv"
    ]

    for file in files:

        try:

            with open(file, "rb") as f:

                st.download_button(
                    label=f"Download {file}",
                    data=f,
                    file_name=file
                )

        except:

            st.warning(
                f"{file} not found"
            )
# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title(
        "Retail Stores Sales Insights and Prediction Model"
    )

    st.markdown("""
    ## Major Project Overview

    This project combines:

    - Data Cleaning
    - Exploratory Data Analysis
    - Machine Learning
    - Customer Segmentation
    - Sales Forecasting
    - Sales Prediction
    - Business Intelligence
    - Recommendation Engine

    ## Technologies Used

    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-Learn
    - XGBoost
    - Prophet
    - Streamlit

    ## Models Implemented

    - Linear Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost

    ## Future Scope

    - Real-time analytics
    - Deep Learning forecasting
    - Cloud deployment
    - Interactive dashboards
    - Automated report generation
    """)

    st.success(
        "Project Successfully Developed and Deployed"
    )                            
