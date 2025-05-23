import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ————— Configuración —————
base_dir    = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'
input_csv   = os.path.join(base_dir, 'client_util_sale_with_state.csv')
output_csv  = os.path.join(base_dir, 'datos_CLIENTES.csv')

# ————— Carga de datos —————
df = pd.read_csv(input_csv)

# Asegurar que 'state' es int
if df['state'].dtype == 'object':
    mapping = {'CORRECTO': 0, 'PENDIENTE_DE_PAGO': 1, 'IMPAGO': 2}
    df['state'] = df['state'].map(mapping)

# Split en 85 % entrenamiento, 15 % validación
_, df_val = train_test_split(
    df[['sale_id', 'state']],
    test_size=0.15,
    stratify=df['state'],
    random_state=42
)

# Guardar CSV
df_val.to_csv(output_csv, index=False)
print(f"CSV guardado en: {output_csv}")
