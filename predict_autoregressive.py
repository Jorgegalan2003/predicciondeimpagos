import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ---------------- CONFIG ----------------
csv_path = r"C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS\client_util_sale_env_3_Panama_proteo_v3.csv"
output_csv = "predicciones_autoregresivas_con_días_Panamá.csv"
model_save_path = "modelo_entrenado.pt"
feature_cols = ["quantity", "days_to_next_sale", "non_payment_period", "amount"]
target_col = "state"
state_map = {"CORRECTO": 0, "PENDIENTE_DE_PAGO": 1, "IMPAGO": 2}
n_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET ----------------
class SalesDataset(Dataset):
    def __init__(self, df):
        self.X = []
        self.y_labels = []
        self.test_rows = []

        for _, group in df.groupby("client_id"):
            group = group.reset_index(drop=True)
            x = torch.tensor(group[feature_cols].values, dtype=torch.float)
            y = torch.tensor(group[target_col].map(state_map).values, dtype=torch.long)
            n = len(x)
            if n < 3:
                continue
            split = int(n * 0.80)
            if split < 1 or (n - split) < 1:
                continue
            self.X.append(x[:split])
            self.y_labels.append(y[split:])
            self.test_rows.append(group.iloc[split:])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_labels[idx], self.test_rows[idx]

# ---------------- MODELO ----------------
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_dim=8):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=embed_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_hist, future_len):
        batch_size = x_hist.size(0)
        device = x_hist.device
        h0 = torch.zeros(1, batch_size, self.encoder.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.encoder.hidden_size).to(device)
        _, (h, c) = self.encoder(x_hist, (h0, c0))

        preds = []
        y_prev = torch.zeros(batch_size, dtype=torch.long).to(device)
        inp = self.embed(y_prev).unsqueeze(1)

        for _ in range(future_len):
            out, (h, c) = self.decoder(inp, (h, c))
            logits = self.fc(out[:, -1, :])
            preds.append(logits)
            y_prev = torch.argmax(logits, dim=1)
            inp = self.embed(y_prev).unsqueeze(1)

        return torch.stack(preds, dim=1).squeeze(0)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(csv_path)
df["state"] = df["state"].replace("PENDIEN", "PENDIENTE_DE_PAGO")
df = df.dropna(subset=feature_cols + [target_col, "sale_id", "client_id"])

# Calcular pesos de clase
y_all = df["state"].map(state_map).dropna().values
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_all)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

dataset = SalesDataset(df)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

# ---------------- TRAIN ----------------
model = AutoregressiveLSTM(input_size=4, hidden_size=128, output_size=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

print("🧠 Comenzando entrenamiento...")

best_loss = float("inf")
patience = 7
trials = 0
n_epochs = 100

for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0
    for x_hist, y_true, _ in dataloader:
        x_hist = x_hist.unsqueeze(0).to(device)
        y_true = y_true.to(device)
        optimizer.zero_grad()
        logits = model(x_hist, future_len=len(y_true))
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:2d} - Loss: {total_loss:.4f}")

    if total_loss < best_loss - 1e-3:
        best_loss = total_loss
        trials = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        trials += 1
        if trials >= patience:
            print(f"⏹️ Early stopping en la epoch {epoch}.")
            break

# ---------------- PREDICT ----------------
model.load_state_dict(torch.load(model_save_path))
model.eval()
results = []
y_true_all = []
y_pred_all = []

with torch.no_grad():
    for x_hist, y_true, test_df in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]):
        x_hist = x_hist.unsqueeze(0).to(device)
        preds = model(x_hist, future_len=len(y_true))
        probs = torch.softmax(preds, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        for idx in range(len(pred_classes)):
            row = test_df.iloc[idx]
            true_label = y_true[idx].item()
            pred_label = pred_classes[idx].item()

            results.append({
                "client_id": row["client_id"],
                "sale_id": row["sale_id"],
                "true": true_label,
                "pred": pred_label,
                "p_CORRECTO": probs[idx, 0].item(),
                "p_PENDIENTE_DE_PAGO": probs[idx, 1].item(),
                "p_IMPAGO": probs[idx, 2].item()
            })

            y_true_all.append(true_label)
            y_pred_all.append(pred_label)

# Guardar predicciones
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n📦 Predicciones guardadas en: {output_csv}")

# ---------------- EVALUACIÓN FINAL ----------------
print("\n🧾 Reporte de clasificación:")
print(classification_report(y_true_all, y_pred_all, target_names=["CORRECTO", "PENDIENTE_DE_PAGO", "IMPAGO"]))

print("📉 Matriz de confusión:")
print(confusion_matrix(y_true_all, y_pred_all))
