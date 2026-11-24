Retail Store Sales Insights and Prediction Model

Project Overview

The **Retail Store Sales Insights and Prediction Model** is a data analytics and machine learning project designed to analyze retail sales data, identify meaningful business insights, understand customer purchasing patterns, and forecast future sales.

The project uses historical Superstore sales data to perform data preprocessing, Exploratory Data Analysis (EDA), customer segmentation, predictive modeling, and time-series forecasting. An interactive dashboard built using Streamlit presents the insights and visualizations in an easy-to-understand format.

Project Objectives

- Analyze historical retail sales and identify sales trends.
- Understand customer purchasing behavior and product performance.
- Identify high-performing and low-performing product categories.
- Segment customers based on their purchasing patterns.
- Build machine learning models to predict sales.
- Forecast future sales using time-series analysis.
- Provide actionable business recommendations through an interactive dashboard.

Technologies & Tools Used


 Programming Language : Python 
 Data Analysis :Pandas, NumPy 
 Data Visualization :Matplotlib, Seaborn, Plotly 
 Machine Learning : Scikit-learn, XGBoost 
 Forecasting :Facebook Prophet 
 Dashboard  Streamlit 
 Development Environment  Jupyter Notebook, VS Code 

 Project Workflow

 1. Data Collection
- Used the Superstore retail sales dataset.
- Dataset contains approximately 9,800 records and 18 columns.

2. Data Cleaning & Preprocessing
- Handled missing values in the dataset.
- Processed date columns and extracted Year, Month, and Day features.
- Handled missing Postal Code values using median imputation.
- Prepared the cleaned dataset for analysis and modeling.

3. Exploratory Data Analysis (EDA)
- Analyzed overall sales and profit distribution.
- Identified monthly and yearly sales trends.
- Evaluated product category and sub-category performance.
- Explored customer purchasing patterns.
- Created visualizations to identify business trends.

4. Machine Learning & Prediction

Implemented and evaluated multiple machine learning algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The models were evaluated using regression performance metrics to understand their predictive capabilities.

5. Customer Segmentation

Applied clustering techniques to group customers based on purchasing patterns and spending behavior.

The segmentation helps identify different customer groups and supports targeted marketing strategies.

6. Sales Forecasting

- Implemented Facebook Prophet for time-series forecasting.
- Analyzed historical sales trends.
- Generated future sales predictions.
- Visualized forecasted sales to support business planning.

7. Interactive Dashboard

Developed an interactive Streamlit dashboard to present:
- Sales performance insights.
- Product and category analysis.
- Customer segmentation results.
- Machine learning predictions.
- Future sales forecasts.

Business Insights & Recommendations

The project helps businesses:

- Identify profitable product categories.
- Understand customer purchasing behavior.
- Improve inventory and sales planning.
- Develop targeted marketing strategies.
- Identify opportunities to improve overall profitability.
- Make data-driven business decisions using sales forecasts.

Installation & Setup

Prerequisites
- Python 3.9 or above
- pip package manager

Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

Step 2: Navigate to the Project Directory

```bash
cd Retail_Store_Analytics_Major_Project
```

Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost prophet streamlit
```

Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

*Note: Replace `app.py` with your actual Streamlit entry-point filename if different.*

Dataset

The project uses the Superstore retail sales dataset containing information about:

- Orders and order dates
- Customer details
- Product categories and sub-categories
- Sales and profit
- Discounts and quantities
- Geographical information

 Future Enhancements

- Deploy the dashboard on AWS.
- Integrate real-time sales data.
- Improve prediction accuracy through feature engineering and model optimization.
- Add advanced customer analytics.
- Implement automated sales reporting.



Data Science | Machine Learning | Python | SQL | Data Analytics

