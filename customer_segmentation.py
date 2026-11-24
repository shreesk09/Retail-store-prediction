import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset

df = pd.read_csv("cleaned_superstore1.csv")

# Customer Sales Summary

customer_data = df.groupby(
    ['Customer ID','Customer Name']
).agg({
    'Sales':'sum'
}).reset_index()

print(customer_data.head())

# Scaling

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    customer_data[['Sales']]
)

# KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

customer_data['Cluster'] = kmeans.fit_predict(
    scaled_data
)

# Cluster Summary

print("\nCluster Summary\n")

print(
    customer_data.groupby('Cluster')['Sales']
    .agg(['count','mean','min','max'])
)

# Visualization

plt.figure(figsize=(8,5))

plt.scatter(
    customer_data.index,
    customer_data['Sales'],
    c=customer_data['Cluster']
)

plt.title("Customer Segmentation")

plt.xlabel("Customers")

plt.ylabel("Total Sales")

plt.show()

# Save

customer_data.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nCustomer Segmentation Completed")