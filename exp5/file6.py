# ================================ 
# STEP 1 : Import Libraries 
# ================================ 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, linkage 
import datetime as dt 
print("Libraries imported successfully") 
# ================================ 
# STEP 2 : Load Dataset 
# ================================ 
df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1') 
print("Dataset Loaded Successfully") 
print(df.head()) 
# ================================ 
# STEP 3 : Data Cleaning 
# ================================ 
# Remove missing CustomerID 
df = df.dropna(subset=['CustomerID']) 
# Remove negative or zero values 
df = df[df['Quantity'] > 0] 
df = df[df['UnitPrice'] > 0] 
print("Data cleaned") 
print(df.shape) 
# ================================ 
# STEP 4 : Create Total Price 
# ================================ 
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
VAIDEHI SETIYA D15B/49 
print("TotalPrice column created") 
print(df.head()) 
# ================================ 
# STEP 5 : Convert Invoice Date 
# ================================ 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True) 
print("Date converted successfully") 
# ================================ 
# STEP 6 : Create RFM Features 
# ================================ 
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1) 
rfm = df.groupby('CustomerID').agg({ 
 'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 
 'InvoiceNo': 'count', 
 'TotalPrice': 'sum' 
}) 
rfm.columns = ['Recency','Frequency','Monetary'] 
print("RFM Table Created") 
print(rfm.head()) 
# ================================ 
# STEP 7 : Feature Scaling 
# ================================ 
scaler = StandardScaler() 
rfm_scaled = scaler.fit_transform(rfm) 
print("Feature scaling completed")
VAIDEHI SETIYA D15B/49 
# ================================ 
# PART A : K-MEANS CLUSTERING 
# ================================ 
# ================================ 
# STEP 8 : Elbow Method 
# ================================ 
wcss = [] 
for i in range(1,11): 
 kmeans = KMeans(n_clusters=i, random_state=42) 
 kmeans.fit(rfm_scaled) 
 wcss.append(kmeans.inertia_) 
plt.figure(figsize=(8,5)) 
plt.plot(range(1,11), wcss, marker='o') 
plt.title("Elbow Method") 
plt.xlabel("Number of Clusters") 
plt.ylabel("WCSS") 
plt.show() 
# ================================ 
# STEP 9 : Apply K-Means 
# ================================ 
kmeans = KMeans(n_clusters=4, random_state=42) 
rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled) 
print("KMeans clustering applied") 
print(rfm.head()) 
# ================================ 
# STEP 10 : Visualize K-Means Clusters 
# ================================ 
plt.figure(figsize=(8,6)) 
plt.scatter( 
 rfm_scaled[:,0], 
 rfm_scaled[:,1], 
 c=rfm['KMeans_Cluster']; 
 cmap='viridis' 
) 
plt.title("Customer Segments using K-Means") 
plt.xlabel("Recency") 
plt.ylabel("Frequency") 
plt.show() 
PART B: HIERARCHICAL CLUSTERING 
# ================================ 
# PART B : HIERARCHICAL CLUSTERING 
# ================================ 
# ================================ 
# STEP 11 : Create Dendrogram 
# ================================ 
linked = linkage(rfm_scaled, method='ward') 
plt.figure(figsize=(10,6)) 
dendrogram(linked) 
plt.title("Dendrogram for Hierarchical Clustering") 
plt.xlabel("Customers") 
plt.ylabel("Euclidean Distance") 
plt.show() 
# ================================ 
# STEP 12 : Apply Agglomerative Clustering 
# ================================ 
hc = AgglomerativeClustering(n_clusters=4, linkage='ward') 
rfm['Hierarchical_Cluster'] = hc.fit_predict(rfm_scaled)
print("Hierarchical clustering applied") 
print(rfm.head()) 
# ================================ 
# STEP 13 : Visualize Hierarchical Clusters 
# ================================ 
plt.figure(figsize=(8,6)) 
plt.scatter( 
 rfm_scaled[:,0], 
 rfm_scaled[:,1], 
 c=rfm['Hierarchical_Cluster'], 
 cmap='rainbow' 
) 
plt.title("Customer Segments using Hierarchical Clustering") 
plt.xlabel("Recency") 
plt.ylabel("Frequency") 
plt.show() 
# ================================ 
# STEP 14 : Cluster Summary 
# ================================ 
cluster_summary = rfm.groupby('KMeans_Cluster').mean() 
print("Cluster Summary") 
print(cluster_summary)

