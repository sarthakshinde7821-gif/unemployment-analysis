import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load dataset
data = pd.read_csv("Unemployment_India.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Convert Date column (fix warning)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

print("Dataset Preview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# -------------------------------
# Overall Unemployment Trend
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(
    data['Date'],
    data['Estimated Unemployment Rate (%)'],
    color='red'
)
plt.title("Unemployment Rate Trend During COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -------------------------------
# Region-wise Unemployment
# -------------------------------
plt.figure(figsize=(12,6))
sns.barplot(
    x='Region',
    y='Estimated Unemployment Rate (%)',
    data=data
)
plt.xticks(rotation=90)
plt.title("Region-wise Unemployment Rate")
plt.show()

# -------------------------------
# Before vs After COVID Analysis
# -------------------------------
before_covid = data[data['Date'] < '2020-03-01']
after_covid = data[data['Date'] >= '2020-03-01']

before_avg = before_covid['Estimated Unemployment Rate (%)'].mean()
after_avg = after_covid['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate:")
print("Before COVID:", round(before_avg, 2))
print("After COVID:", round(after_avg, 2))

# Comparison Graph
labels = ['Before COVID', 'After COVID']
values = [before_avg, after_avg]

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.title("Impact of COVID-19 on Unemployment")
plt.ylabel("Unemployment Rate (%)")
plt.show()

print("\nConclusion:")
print("- COVID-19 caused a sharp increase in unemployment.")
print("- Lockdown severely affected employment across regions.")
print("- Data analysis helps understand economic impact clearly.")

# -------------------------------
# ML Prediction using Linear Regression
# -------------------------------

from sklearn.linear_model import LinearRegression
import numpy as np

# Convert Date to ordinal (numeric form)
data_ml = data.copy()
data_ml['Date_Ordinal'] = data_ml['Date'].map(pd.Timestamp.toordinal)

X = data_ml[['Date_Ordinal']]
y = data_ml['Estimated Unemployment Rate (%)']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 6 months
future_dates = pd.date_range(
    start=data['Date'].max(),
    periods=6,
    freq='M'
)

future_ordinals = np.array(
    [d.toordinal() for d in future_dates]
).reshape(-1, 1)

predictions = model.predict(future_ordinals)

# Show predictions
print("\nFuture Unemployment Predictions:")
for d, p in zip(future_dates, predictions):
    print(d.date(), "->", round(p, 2), "%")

# Plot prediction
plt.figure(figsize=(10,5))
plt.plot(data['Date'], y, label="Actual Data")
plt.plot(future_dates, predictions, label="Predicted", linestyle="--")
plt.legend()
plt.title("Unemployment Rate Prediction")
plt.show()

