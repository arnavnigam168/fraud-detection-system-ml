import pandas as pd
import numpy as np

np.random.seed(42)

N = 15000

data = {
    "amount": np.round(np.random.exponential(scale=2000, size=N), 2),
    "transaction_hour": np.random.randint(0, 24, size=N),
    "transaction_type": np.random.choice([0, 1, 2], size=N),
    "merchant_risk": np.random.uniform(0, 1, size=N),
    "device_trust": np.random.choice([0, 1], size=N, p=[0.3, 0.7]),
    "location_change": np.random.choice([0, 1], size=N, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)

fraud_probability = (
    (df["amount"] > 5000).astype(int) * 0.3 +
    (df["transaction_hour"].between(0, 5)).astype(int) * 0.2 +
    (df["merchant_risk"] > 0.7).astype(int) * 0.3 +
    (df["device_trust"] == 0).astype(int) * 0.2 +
    (df["location_change"] == 1).astype(int) * 0.3
)

df["is_fraud"] = (fraud_probability > 0.5).astype(int)

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("data/transactions.csv", index=False)

print("✅ Synthetic fraud dataset generated at data/transactions.csv")
print(df["is_fraud"].value_counts())
