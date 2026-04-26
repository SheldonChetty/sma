import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("Superstore.csv", encoding='latin1')   # change filename if needed

print("First 5 rows:\n", df.head())

# ================= EXPLORATORY ANALYSIS =================
col = df['Sales']

print("\n===== EXPLORATORY ANALYSIS (Sales) =====")
print("1. Mean:", col.mean())
print("2. Median:", col.median())
print("3. Mode:", col.mode()[0])
print("4. Variance:", col.var())
print("5. Standard Deviation:", col.std())
print("6. Minimum:", col.min())
print("7. Maximum:", col.max())
print("8. Range:", col.max() - col.min())
print("9. Skewness:", col.skew())
print("10. Kurtosis:", col.kurt())

# ================= VISUALIZATION =================
print("\n===== VISUALIZATION =====")

# 1 Scatter Plot — Sales vs Profit
plt.figure()
plt.scatter(df['Sales'], df['Profit'])
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Sales vs Profit")
plt.tight_layout()
plt.show()

# 2 Histogram — Sales Distribution
plt.figure()
plt.hist(df['Sales'], bins=10)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.tight_layout()
plt.show()

# 3 Pie Chart — Category Distribution
plt.figure()
df['Category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Category Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 4 Line Plot — Sales over Orders
sorted_df = df.sort_values(by="Order Date")
plt.figure()
plt.plot(sorted_df['Sales'])
plt.xlabel("Order Index")
plt.ylabel("Sales")
plt.title("Sales Trend")
plt.tight_layout()
plt.show()

# 5 Box Plot — Profit by Category
plt.figure()
sns.boxplot(x='Category', y='Profit', data=df)
plt.title("Profit by Category")
plt.tight_layout()
plt.show()

# 6 Bar Plot — Sales by Region
plt.figure()
df.groupby('Region')['Sales'].sum().plot(kind='bar')
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Sales by Region")
plt.tight_layout()
plt.show()

# 7 Heatmap — Correlation
plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 8 Area Chart — Sales vs Profit
sorted_df = df.sort_values(by="Sales")
plt.figure()
plt.fill_between(sorted_df['Sales'], sorted_df['Profit'], alpha=0.3)
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Sales vs Profit Area")
plt.tight_layout()
plt.show()

# 9 Violin Plot — Sales Distribution
plt.figure()
sns.violinplot(y=df['Sales'])
plt.title("Sales Distribution Shape")
plt.tight_layout()
plt.show()

# 10 Density Plot — Profit Density
plt.figure()
sns.kdeplot(df['Profit'], fill=True)
plt.title("Profit Density")
plt.tight_layout()
plt.show()