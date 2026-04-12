import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


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

# scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# scale target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# convert to pytorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


# train and evaluate the neural network
def run_NN_model(learning_rate, epochs):
    # create model
    model = SimpleNN(input_size=X_train_tensor.shape[1])

    # loss function
    loss_fn = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(epochs):
        model.train()

        preds = model(X_train_tensor)
        loss = loss_fn(preds, y_train_tensor)

        # save the model and scalers 
        torch.save(model.state_dict(), "models/nn_model.pt")
        joblib.dump(scaler, "models/x_scaler.pkl")
        joblib.dump(y_scaler, "models/y_scaler.pkl")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        # test_preds = model(X_test_tensor).numpy().flatten()
        test_preds_scaled = model(X_test_tensor).numpy()
        test_preds = y_scaler.inverse_transform(test_preds_scaled).flatten()

    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    # save sample predictions
    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": test_preds
    })
    results["error"] = results["actual"] - results["predicted"]

    return mae, rmse, results


# several simple settings to test
# settings = [
#     {"learning_rate": 0.01, "epochs": 50},
#     {"learning_rate": 0.001, "epochs": 50},
#     {"learning_rate": 0.001, "epochs": 100}
# ]

# leaving only best setting based on nn_results.txt output
settings = [
    {"learning_rate": 0.01, "epochs": 50}
]

# save results
with open("nn_results_final.txt", "w") as f:
    f.write("neural network results\n")
    f.write("========================\n\n")

    for setting in settings:
        lr = setting["learning_rate"]
        epochs = setting["epochs"]

        mae, rmse, results = run_NN_model(lr, epochs)

        f.write(f"learning_rate: {lr}, epochs: {epochs}\n")
        f.write(f"mae: {mae}\n")
        f.write(f"rmse: {rmse}\n")
        f.write("sample predictions:\n")
        f.write(results.head(10).to_string())
        f.write("\n\n")
