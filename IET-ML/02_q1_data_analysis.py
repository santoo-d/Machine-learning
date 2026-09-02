import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
raw = pd.read_csv("data/air_quality_raw.csv", parse_dates=["Date"])

print("=== RAW DATA INFO ===")
print("Shape:", raw.shape)
print("Duplicates:", raw.duplicated().sum())
print("Missing values per column:\n", raw.isna().sum())

# --- Cleaning ---
df = raw.drop_duplicates().copy()
df = df.sort_values("Date").reset_index(drop=True)

num_cols = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
# Impute missing numeric values with median (robust to outliers)
for c in num_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

# Outlier capping using IQR (winsorize)
for c in num_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    df[c] = df[c].clip(lo, hi)

# Recompute AQI/labels after cleaning (in case clipping changed values)
df["AQI"] = (df["PM2.5"]*0.5 + df["PM10"]*0.25 + df["NO2"]*0.5 + df["SO2"]*0.6 +
             df["CO"]*20 + df["O3"]*0.3)
q1v, q2v, q3v = df["AQI"].quantile([0.25, 0.60, 0.90])
def classify(aqi):
    if aqi <= q1v: return "Good"
    elif aqi <= q2v: return "Moderate"
    elif aqi <= q3v: return "Poor"
    else: return "Hazardous"
df["AQI_Category"] = df["AQI"].apply(classify)
def health_risk(row):
    if row["AQI_Category"] == "Hazardous": return "High"
    if row["AQI_Category"] == "Poor" and row["PM2.5"] > 140: return "High"
    if row["AQI_Category"] == "Poor": return "Moderate"
    if row["AQI_Category"] == "Moderate": return "Low"
    return "Minimal"
df["HealthRisk"] = df.apply(health_risk, axis=1)

df.to_csv("data/air_quality_cleaned.csv", index=False)

print("\n=== CLEANED DATA ===")
print("Shape after cleaning:", df.shape)
print("\nStatistical summary:\n", df[num_cols+["AQI"]].describe().round(2))

print("\nClass distribution:\n", df["AQI_Category"].value_counts())

corr = df[num_cols+["AQI"]].corr()
print("\nCorrelation with AQI:\n", corr["AQI"].sort_values(ascending=False))

# ---------------- Visualizations (>=5) ----------------
# 1. Correlation heatmap
plt.figure(figsize=(9,7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Air Quality Parameters")
plt.tight_layout(); plt.savefig("figs/01_correlation_heatmap.png", dpi=140); plt.close()

# 2. Distribution of PM2.5
plt.figure(figsize=(7,5))
sns.histplot(df["PM2.5"], kde=True, color="steelblue", bins=30)
plt.title("Distribution of PM2.5 Concentration")
plt.xlabel("PM2.5 (µg/m³)")
plt.tight_layout(); plt.savefig("figs/02_pm25_distribution.png", dpi=140); plt.close()

# 3. AQI category counts
plt.figure(figsize=(7,5))
order = ["Good","Moderate","Poor","Hazardous"]
sns.countplot(x="AQI_Category", data=df, order=order, palette="YlOrRd")
plt.title("Air Quality Category Distribution")
plt.tight_layout(); plt.savefig("figs/03_aqi_category_counts.png", dpi=140); plt.close()

# 4. Boxplot of pollutants by category
plt.figure(figsize=(9,5))
sns.boxplot(x="AQI_Category", y="PM2.5", data=df, order=order, palette="Blues")
plt.title("PM2.5 Levels across Air Quality Categories")
plt.tight_layout(); plt.savefig("figs/04_pm25_by_category_boxplot.png", dpi=140); plt.close()

# 5. Time series of AQI over time (rolling mean)
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["AQI"], alpha=0.35, label="Daily AQI")
plt.plot(df["Date"], df["AQI"].rolling(14).mean(), color="crimson", label="14-day rolling mean")
plt.title("AQI Trend Over Time")
plt.xlabel("Date"); plt.ylabel("Composite AQI")
plt.legend()
plt.tight_layout(); plt.savefig("figs/05_aqi_timeseries.png", dpi=140); plt.close()

# 6. Scatter: WindSpeed vs PM2.5 (negative relationship)
plt.figure(figsize=(7,5))
sns.scatterplot(x="WindSpeed", y="PM2.5", hue="AQI_Category", data=df, hue_order=order, palette="viridis", alpha=0.7)
plt.title("Wind Speed vs PM2.5 (colored by AQI Category)")
plt.tight_layout(); plt.savefig("figs/06_windspeed_vs_pm25.png", dpi=140); plt.close()

print("\nSaved 6 visualizations to figs/")
