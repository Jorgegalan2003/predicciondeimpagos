import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

# Cargar el archivo
df = pd.read_csv("ventas_con_payment_period_CLIENTES.csv")

# Variables
features = ['quantity', 'amount', 'rate_payment_period',
            'non_payment_period', 'community_id']
target = 'state'
meta_columns = ['client_id', 'sale_id', 'environment_id']

# Filtrar columnas necesarias y eliminar nulos
df_model = df[features + [target] + meta_columns].dropna()

# Codificar variable objetivo
label_encoder = LabelEncoder()
df_model[target] = label_encoder.fit_transform(df_model[target])

# One-hot encoding de community_id
df_model = pd.get_dummies(df_model, columns=['community_id'], prefix='community')

# Separar conjunto por cliente: 80/20
train_data = []
test_data = []

for client_id, group in df_model.groupby('client_id'):
    group = shuffle(group, random_state=42)
    split_idx = int(len(group) * 0.8)
    train_data.append(group.iloc[:split_idx])
    test_data.append(group.iloc[split_idx:])

train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

# Separar datos de entrada, salida y metadatos
X_train = train_df.drop(columns=[target] + meta_columns)
y_train = train_df[target]
X_test = test_df.drop(columns=[target] + meta_columns)
y_test = test_df[target]
meta_test = test_df[meta_columns]

# Modelo con ajuste por desbalance
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Reporte
print("REPORTE DE CLASIFICACIÓN POR CLIENTE (con balance y community_id categórica):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("MATRIZ DE CONFUSIÓN (Por Cliente):")
print(confusion_matrix(y_test, y_pred))

# Guardar CSV con predicciones
pred_df = meta_test.copy()
pred_df['real'] = label_encoder.inverse_transform(y_test)
pred_df['predicho'] = label_encoder.inverse_transform(y_pred)
pred_df.to_csv("predicciones_por_cliente_completo.csv", index=False)
print("Archivo 'predicciones_por_cliente_completo.csv' guardado.")
