import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("cleaned_superstore1.csv")

# Convert date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Monthly Sales
monthly_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Prophet format
monthly_sales.columns = ['ds', 'y']

# Create model
model = Prophet()

model.fit(monthly_sales)

# Future Forecast
future = model.make_future_dataframe(
    periods=365
)

forecast = model.predict(future)

# Display forecast
print(
    forecast[
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].tail()
)

# Plot forecast
fig1 = model.plot(forecast)

plt.title("Future Sales Forecast")

plt.show()

# Components
fig2 = model.plot_components(forecast)

plt.show()

# Save
forecast.to_csv(
    "sales_forecast.csv",
    index=False
)

print("\nForecasting Completed Successfully")