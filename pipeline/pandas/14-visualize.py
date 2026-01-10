# Remove Weighted_Price
df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp -> Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert timestamp to datetime and index on Date
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")

# Fill missing values
df["Close"] = df["Close"].ffill()

for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])

df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Keep data from 2017 onwards
df = df.loc["2017-01-01":]

# Resample daily with required aggregations
df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum",
})

# Reorder columns to match expected output
df = df[["High", "Low", "Open", "Close", "Volume_(BTC)", "Volume_(Currency)"]]

# Rename index to Date and display the transformed df (before plotting)
df.index.name = "Date"
print(df)

# Plot
df.plot()
plt.show()
