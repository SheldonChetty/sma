import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Books2.csv")

print("First 5 rows:\n", df.head())

# ================= EXPLORATORY ANALYSIS =================
col = df['actual_productivity_score']

print("\n===== EXPLORATORY ANALYSIS =====")
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

# 1 Scatter Plot
plt.figure()
plt.scatter(df['daily_social_media_time'], df['actual_productivity_score'])
plt.xlabel("Daily Social Media Time")
plt.ylabel("Productivity Score")
plt.title("Social Media Time vs Productivity")
plt.tight_layout()
plt.show()

# 2 Histogram
plt.figure()
plt.hist(df['actual_productivity_score'], bins=5)
plt.title("Productivity Distribution")
plt.xlabel("Productivity Score")
plt.tight_layout()
plt.show()

# 3 Pie Chart
plt.figure()
df['social_platform_preference'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Platform Preference")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 4 Line Plot
sorted_df = df.sort_values(by="work_hours_per_day")
plt.figure()
plt.plot(sorted_df['work_hours_per_day'], sorted_df['breaks_during_work'])
plt.xlabel("Work Hours")
plt.ylabel("Breaks")
plt.title("Work Hours vs Breaks")
plt.tight_layout()
plt.show()

# 5 Box Plot
plt.figure()
sns.boxplot(x='job_type', y='actual_productivity_score', data=df)
plt.title("Productivity by Job Type")
plt.tight_layout()
plt.show()

# 6 Bar Plot
df['sleep_group'] = pd.cut(df['sleep_hours'], bins=4)
plt.figure()
df.groupby('sleep_group', observed=True)['coffee_consumption_per_day'].mean().plot(kind='bar')
plt.xlabel("Sleep Hours Group")
plt.ylabel("Avg Coffee Consumption")
plt.title("Sleep vs Coffee")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7 Heatmap
plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 8 Area Chart
sorted_df = df.sort_values(by="stress_level")
plt.figure()
plt.fill_between(sorted_df['stress_level'], sorted_df['actual_productivity_score'], alpha=0.3)
plt.xlabel("Stress Level")
plt.ylabel("Productivity")
plt.title("Stress vs Productivity")
plt.tight_layout()
plt.show()

# 9 Violin Plot
plt.figure()
sns.violinplot(y=df['actual_productivity_score'])
plt.title("Productivity Distribution")
plt.tight_layout()
plt.show()

# 10 Density Plot
plt.figure()
sns.kdeplot(df['daily_social_media_time'], fill=True)
plt.title("Social Media Time Density")
plt.tight_layout()
plt.show()