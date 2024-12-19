import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Mall_Customers.csv")


print(data.head())


X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
range_clusters = range(1, 11)  
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.show()


k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)


data['Cluster'] = kmeans.labels_


plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Annual Income (k$)'], 
                y=data['Spending Score (1-100)'], 
                hue=data['Cluster'], 
                palette='viridis', 
                s=100)
plt.title("Customer Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend(title='Cluster')
plt.show()
