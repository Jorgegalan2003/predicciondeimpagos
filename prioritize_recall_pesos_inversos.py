import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
csv_path = r"C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS\client_util_sale_env_8_Republica_Dominicana.csv"
output_csv = "predicciones_priorizando_recall_pesos_inversos_RD.csv"
model_save_path = "modelo_priorizando_recall.pt"
feature_cols = ["quantity", "rate_id", "non_payment_period", "amount"]
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

# Separar clientes en train y validación
client_ids = df["client_id"].unique()
train_ids, val_ids = train_test_split(client_ids, test_size=0.2, random_state=42)

train_df = df[df["client_id"].isin(train_ids)].copy()
val_df = df[df["client_id"].isin(val_ids)].copy()

train_dataset = SalesDataset(train_df)
val_dataset = SalesDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

# ---------------- CLASS WEIGHTS ----------------
# Calcular pesos inversos a la frecuencia de aparición en el dataset de entrenamiento
class_counts = train_df[target_col].map(state_map).value_counts().sort_index()
class_freq = class_counts / class_counts.sum()
inverse_freq_weights = 1.0 / class_freq
weights = torch.tensor(inverse_freq_weights.values, dtype=torch.float32).to(device)

print("Pesos por clase (inversos a la frecuencia):", weights)

# ---------------- TRAIN ----------------
model = AutoregressiveLSTM(input_size=4, hidden_size=64, output_size=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=weights)

print("🧠 Comenzando entrenamiento (optimizando recall IMPAGO)...")

best_recall = 0.0
trials = 0
patience = 8
n_epochs = 100

for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0
    for x_hist, y_true, _ in train_loader:
        x_hist = x_hist.unsqueeze(0).to(device)
        y_true = y_true.to(device)
        optimizer.zero_grad()
        logits = model(x_hist, future_len=len(y_true))
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # VALIDACIÓN
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x_hist, y_true, _ in val_loader:
            x_hist = x_hist.unsqueeze(0).to(device)
            y_true = y_true.to(device)
            logits = model(x_hist, future_len=len(y_true))
            preds = torch.argmax(logits, dim=1)
            all_true.extend(y_true.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    # Calcular recall de IMPAGO
    true_impago = [y for y in all_true if y == 2]
    pred_impago = [p for y, p in zip(all_true, all_pred) if y == 2]
    recall_impago = recall_score(true_impago, pred_impago, average='macro', zero_division=0)

    print(f"Epoch {epoch:2d} - Loss: {total_loss:.4f} - Recall IMPAGO: {recall_impago:.4f}")

    if recall_impago > best_recall + 1e-4:
        best_recall = recall_impago
        trials = 0
        torch.save(model.state_dict(), model_save_path)
        print("✅ Modelo guardado con mejor recall IMPAGO.")
    else:
        trials += 1
        if trials >= patience:
            print(f"⏹️ Early stopping en la epoch {epoch}.")
            break

# ---------------- PREDICT ----------------
model.load_state_dict(torch.load(model_save_path))
model.eval()
results = []
all_true = []
all_pred = []

with torch.no_grad():
    for x_hist, y_true, test_df in DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]):
        x_hist = x_hist.unsqueeze(0).to(device)
        preds = model(x_hist, future_len=len(y_true))
        probs = torch.softmax(preds, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        for idx in range(len(pred_classes)):
            row = test_df.iloc[idx]
            results.append({
                "client_id": row["client_id"],
                "sale_id": row["sale_id"],
                "true": y_true[idx].item(),
                "pred": pred_classes[idx].item(),
                "p_CORRECTO": probs[idx, 0].item(),
                "p_PENDIENTE_DE_PAGO": probs[idx, 1].item(),
                "p_IMPAGO": probs[idx, 2].item()
            })
            all_true.append(y_true[idx].item())
            all_pred.append(pred_classes[idx].item())

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n📦 Predicciones guardadas en: {output_csv}")

# ---------------- EVALUACIÓN FINAL ----------------
print("\n🧾 Reporte de clasificación (validación):")
print(classification_report(all_true, all_pred, target_names=["CORRECTO", "PENDIENTE_DE_PAGO", "IMPAGO"]))
print("Matriz de confusión:")
print(confusion_matrix(all_true, all_pred))
