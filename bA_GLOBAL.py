import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Cargar datos
df = pd.read_csv("ventas_con_payment_period_CLIENTES.csv")

# Variables
features = ['quantity', 'amount', 'rate_payment_period',
            'non_payment_period', 'community_id']
target = 'state'
meta_columns = ['client_id', 'sale_id', 'environment_id']  # columnas a mantener para el CSV final

# Eliminar nulos
df_model = df[features + [target] + meta_columns].dropna()

# Codificar variable objetivo
label_encoder = LabelEncoder()
df_model[target] = label_encoder.fit_transform(df_model[target])

# One-hot encoding de community_id
df_model = pd.get_dummies(df_model, columns=['community_id'], prefix='community')

# Separar X, y y columnas extra
X = df_model.drop(columns=[target] + meta_columns)
y = df_model[target]
meta = df_model[meta_columns]

# División train/test global
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)

# Entrenar modelo con balance de clases
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Reporte
print("REPORTE DE CLASIFICACIÓN GLOBAL (con balance y community_id categórica):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("MATRIZ DE CONFUSIÓN (Global):")
print(confusion_matrix(y_test, y_pred))

# Crear CSV con predicciones
pred_df = meta_test.copy()
pred_df['real'] = label_encoder.inverse_transform(y_test)
pred_df['predicho'] = label_encoder.inverse_transform(y_pred)
pred_df.to_csv("predicciones_global_completo.csv", index=False)
print("Archivo 'predicciones_global_completo.csv' guardado.")
