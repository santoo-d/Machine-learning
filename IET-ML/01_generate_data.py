import numpy as np, pandas as pd

np.random.seed(42)
N = 1200

dates = pd.date_range("2024-01-01", periods=N, freq="D")

# Seasonal temperature (deg C) with yearly cycle
day_of_year = dates.dayofyear.values
temp = 26 + 8*np.sin(2*np.pi*(day_of_year-80)/365) + np.random.normal(0,2.5,N)
humidity = np.clip(55 - 0.6*(temp-26) + np.random.normal(0,8,N), 10, 100)
wind_speed = np.clip(np.random.gamma(2.0, 1.4, N), 0.2, 15)
pressure = 1013 + np.random.normal(0,4,N) - 0.3*(temp-26)

# Pollutants: higher when wind low, humidity high, winter months (temp low) -> inverse relation
base_pm25 = 120 - 3.2*wind_speed - 0.9*temp + 0.35*humidity + np.random.normal(0,18,N)
pm25 = np.clip(base_pm25, 5, 400)
pm10 = np.clip(pm25*1.6 + np.random.normal(0,15,N), 10, 500)
no2 = np.clip(25 + 0.25*pm25 - 1.1*wind_speed + np.random.normal(0,8,N), 2, 200)
so2 = np.clip(10 + 0.08*pm25 - 0.4*wind_speed + np.random.normal(0,4,N), 1, 80)
co = np.clip(0.4 + 0.012*pm25 - 0.02*wind_speed + np.random.normal(0,0.3,N), 0.1, 6)
o3 = np.clip(30 + 0.6*temp - 0.15*humidity + np.random.normal(0,10,N), 2, 180)

df = pd.DataFrame({
    "Date": dates, "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
    "CO": co, "O3": o3, "Temperature": temp, "Humidity": humidity,
    "WindSpeed": wind_speed, "Pressure": pressure
})

# Compute a composite AQI-like score (simplified, weighted) to derive class labels
df["AQI"] = (df["PM2.5"]*0.5 + df["PM10"]*0.25 + df["NO2"]*0.5 + df["SO2"]*0.6 +
             df["CO"]*20 + df["O3"]*0.3)

q1, q2, q3 = df["AQI"].quantile([0.25, 0.60, 0.90])

def classify(aqi):
    if aqi <= q1: return "Good"
    elif aqi <= q2: return "Moderate"
    elif aqi <= q3: return "Poor"
    else: return "Hazardous"

df["AQI_Category"] = df["AQI"].apply(classify)

# Health risk flag: Hazardous or Poor with high PM2.5 => High risk
def health_risk(row):
    if row["AQI_Category"] == "Hazardous": return "High"
    if row["AQI_Category"] == "Poor" and row["PM2.5"] > 140: return "High"
    if row["AQI_Category"] in ("Poor",): return "Moderate"
    if row["AQI_Category"] == "Moderate": return "Low"
    return "Minimal"

df["HealthRisk"] = df.apply(health_risk, axis=1)

# introduce some missingness and duplicate rows to simulate real-world dirty data
miss_idx = np.random.choice(df.index, size=40, replace=False)
for i in miss_idx:
    col = np.random.choice(["PM2.5","NO2","Humidity","WindSpeed"])
    df.loc[i, col] = np.nan

dup_rows = df.sample(15, random_state=1)
df_dirty = pd.concat([df, dup_rows], ignore_index=True)

df_dirty.to_csv("data/air_quality_raw.csv", index=False)
df.to_csv("data/air_quality_clean_reference.csv", index=False)
print("Raw shape:", df_dirty.shape)
print(df["AQI_Category"].value_counts())
print(df["HealthRisk"].value_counts())
