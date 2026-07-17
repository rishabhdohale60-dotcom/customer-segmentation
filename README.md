# 🛍️ Customer Segmentation

Unsupervised Machine Learning project that segments mall customers into 5 distinct groups using KMeans Clustering.

## 📊 Dataset
- 200 mall customer records
- Source: Kaggle - Mall Customer Segmentation

## 🔍 Features Used
- Annual Income (k$)
- Spending Score (1-100)

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib

## 🎯 5 Customer Segments Found
| Cluster | Income | Spending | Type |
|---------|--------|----------|------|
| 🔴 0 | Medium | Medium | Average Customers |
| 🔵 1 | High | High | VIP Customers 💰 |
| 🟢 2 | Low | High | Impulse Buyers 🛍️ |
| 🟡 3 | High | Low | Potential Targets 🎯 |
| 🟣 4 | Low | Low | Budget Conscious 😐 |

## 🔄 ML Pipeline
1. Data Loading & Exploration
2. Feature Selection
3. StandardScaler
4. Elbow Method (optimal clusters = 5)
5. KMeans Clustering
6. Visualization

## 💡 Key Learnings
- Unsupervised Learning — no labels needed!
- Elbow Method to find optimal clusters
- KMeans groups similar customers together
- Real business insights from data!
