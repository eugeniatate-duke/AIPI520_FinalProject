import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# load data
df = pd.read_csv("data/processed/features_data.csv")

# select features 
features = [
    "MONTH",
    "YEAR",
    "TOTALCUSTOMERS",
    "prev_m_usage",
    "rolling_3_avg",
    "kwh_per_customer"
]

# target 
target = "TOTALKWH"

X = df[features]
y = df[target]

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# get predictions
preds = model.predict(X_test)

# evaluate MAE and RMSE 
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("mae:", mae)
print("rmse:", rmse)