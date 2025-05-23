import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------- CONFIG ----------------
csv_path = "ventas_con_payment_period_Filipinas.csv"
output_csv = "errores_relativos_Filipinas_caso1.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = ["quantity", "days_to_next_sale", "non_payment_period", "amount"]

# ---------------- DATASET ----------------
class SalesDataset(Dataset):
    def __init__(self, df):
        self.X = []
        self.y_days = []
        self.test_rows = []

        df["payment_date"] = pd.to_datetime(df["payment_date"])

        for _, group in df.groupby("client_id"):
            group = group.sort_values("payment_date").reset_index(drop=True)
            if len(group) < 3:
                continue

            group["days_until_next"] = (group["payment_date"].shift(-1) - group["payment_date"]).dt.days
            group = group.dropna(subset=feature_cols + ["days_until_next", "rate_payment_period"])

            n = len(group)
            split = int(n * 0.80)
            if split < 1 or (n - split) < 1:
                continue

            x = torch.tensor(group[feature_cols].values[:split], dtype=torch.float)
            y = torch.tensor(group["days_until_next"].values[split:], dtype=torch.float)

            self.X.append(x)
            self.y_days.append(y)
            self.test_rows.append(group.iloc[split:].copy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_days[idx], self.test_rows[idx]

# ---------------- MODELO ----------------
class LSTMDaysPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x_hist, future_len):
        batch_size = x_hist.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x_hist.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x_hist.device)
        _, (h, c) = self.lstm(x_hist, (h0, c0))

        preds = []
        inp = x_hist[:, -1:, :]

        for _ in range(future_len):
            out, (h, c) = self.lstm(inp, (h, c))
            days_pred = self.fc(out[:, -1, :])
            preds.append(days_pred.squeeze(1))
            inp = inp  # dummy autoregresión

        return torch.cat([p.unsqueeze(1) for p in preds], dim=1)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(csv_path)
df = df.dropna(subset=feature_cols + ["sale_id", "client_id", "payment_date", "rate_payment_period"])

dataset = SalesDataset(df)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

# ---------------- TRAIN ----------------
model = LSTMDaysPredictor(
    input_size=len(feature_cols),
    hidden_size=64
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

n_epochs = 50
for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0

    for x_hist, y_days, _ in dataloader:
        x_hist = x_hist.unsqueeze(0).to(device)
        y_days = y_days.to(device)

        optimizer.zero_grad()
        preds = model(x_hist, future_len=y_days.size(0))
        loss = criterion(preds.squeeze(0), y_days)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:2d} - Loss: {total_loss:.4f}")

# ---------------- PREDICT & SAVE ERRORS ----------------
model.eval()
total_preds = 0
total_relative_error = 0.0
results = []

with torch.no_grad():
    for x_hist, y_days, test_df in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]):
        x_hist = x_hist.unsqueeze(0).to(device)
        preds = model(x_hist, future_len=y_days.size(0)).squeeze(0).cpu()

        errors = torch.abs(preds - y_days.cpu())
        rate_periods = test_df["rate_payment_period"].values.astype(float)
        denominators = rate_periods * 3
        relative_errors = errors / torch.tensor(denominators, dtype=torch.float)

        total_preds += len(relative_errors)
        total_relative_error += relative_errors.sum().item()

        for idx in range(len(errors)):
            row = test_df.iloc[idx]
            results.append({
                "client_id": row["client_id"],
                "sale_id": row["sale_id"],
                "rate_payment_period": row["rate_payment_period"],
                "real_days_until_next": y_days[idx].item(),
                "predicted_days": preds[idx].item(),
                "abs_error_days": errors[idx].item(),
                "relative_error": relative_errors[idx].item()
            })

# ---------------- RESULTADOS ----------------
avg_relative_error = total_relative_error / total_preds
print(f"\n📊 Error relativo promedio: {avg_relative_error:.4f}")

# ---------------- CSV CON ERRORES ----------------
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"📦 Archivo guardado: {output_csv}")
