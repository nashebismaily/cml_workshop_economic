import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("economic_data.csv")

# Set style for plots
sns.set_theme(style="whitegrid")

# Pairplot to visualize relationships between features
sns.pairplot(data, diag_kind='kde', corner=True)
plt.suptitle("Economic Indicators - Pairwise Relationships", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Economic Indicators")
plt.show()

# Boxplot for loan default risk
plt.figure(figsize=(8, 4))
sns.boxplot(x=data["loan_default_risk"])
plt.title("Loan Default Risk Distribution")
plt.show()

# Summary statistics
print(data.describe())
