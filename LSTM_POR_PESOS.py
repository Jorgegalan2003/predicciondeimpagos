import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ————— CONFIG —————
base_dir    = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'
country_csv = os.path.join(base_dir, 'client_util_sale_env_8_Republica_Dominicana.csv')
model_dir   = os.path.join(base_dir, 'model_single_country_pytorch')
preds_dir   = os.path.join(base_dir, 'preds_single_country_pytorch')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(preds_dir, exist_ok=True)

feature_cols = ['quantity', 'rate_id', 'non_payment_period', 'amount']
target_col   = 'state'
n_classes    = 3
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ————— DATASET —————
class SaleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ————— MODEL —————
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.bn   = nn.BatchNorm1d(hidden_dim)
        self.fc1  = nn.Linear(hidden_dim, 32)
        self.fc2  = nn.Linear(32, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.drop(out)
        out    = self.bn(out)
        out    = torch.relu(self.fc1(out))
        out    = self.drop(out)
        return self.fc2(out)

# ————— 1) LOAD & PREPROCESS —————
df = pd.read_csv(country_csv)

# Corregir valores erróneos en 'state'
df['state'] = df['state'].replace('PENDIEN', 'PENDIENTE_DE_PAGO')

# Asegura que 'state' es numérico
if df['state'].dtype == 'object':
    mapping = {'CORRECTO': 0, 'PENDIENTE_DE_PAGO': 1, 'IMPAGO': 2}
    df['state'] = df['state'].map(mapping)

X   = df[feature_cols].values
y   = df['state'].values.astype(int)
idx = np.arange(len(df))

# ————— 2) SPLIT TRAIN / VAL —————
X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
    X, y, idx, test_size=0.15, random_state=42, stratify=y
)

# ————— 3) SCALE —————
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)

# ————— 4) DATA LOADERS —————
train_ds = SaleDataset(X_tr_s.reshape(-1, 1, X_tr_s.shape[1]), y_tr)
val_ds   = SaleDataset(X_val_s.reshape(-1, 1, X_val_s.shape[1]), y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)

# ————— 5) MODELO Y ENTRENAMIENTO —————
model = LSTMClassifier(n_features=len(feature_cols)).to(device)

# Calcular pesos de clase con control de errores
class_counts = Counter(y_tr)
print("Distribución de clases en y_tr:", class_counts)

total = sum(class_counts.values())
weights = []
for i in range(n_classes):
    count = class_counts.get(i, 0)
    if count == 0:
        print(f"⚠️ Advertencia: La clase {i} no está en el conjunto de entrenamiento. Se asignará peso 0.")
        weights.append(0.0)
    else:
        weights.append(total / count)

weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float('inf')
patience  = 5
trials    = 0

for epoch in range(1, 51):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        trials = 0
        torch.save(model.state_dict(), os.path.join(model_dir, 'lstm_single_country.pt'))
        with open(os.path.join(model_dir, 'scaler_single_country.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# ————— 6) PREDICCIÓN Y GUARDADO DE CSV —————
model.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_single_country.pt')))
model.eval()
X_val_tensor = torch.tensor(X_val_s.reshape(-1, 1, X_val_s.shape[1]), dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model(X_val_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy() * 100

sale_ids_val = df['sale_id'].iloc[idx_val].values

preds_df = pd.DataFrame({
    'sale_id':             sale_ids_val,
    'p_CORRECTO':          probs[:, 0],
    'p_PENDIENTE_DE_PAGO': probs[:, 1],
    'p_IMPAGO':            probs[:, 2],
    'pred_state':          probs.argmax(axis=1)
})

output_path = os.path.join(preds_dir, 'predicciones_pesos_RD.csv')
preds_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
