import os
import pandas as pd

# Configura tu directorio de trabajo
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'

# Nombre del CSV intermedio que ya contiene non_payment_period
input_csv       = 'client_util_sale_with_npp.csv'
# Fichero original de saleItems con la columna 'amount'
sale_items_file = '_saleItems__202504221841.csv'

# Construye rutas absolutas
input_path      = os.path.join(base_dir, input_csv)
sale_items_path = os.path.join(base_dir, sale_items_file)

# Verifica que los archivos existen
for path in (input_path, sale_items_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontrado: {path}")

# Carga el CSV intermedio
df = pd.read_csv(input_path)

# Carga saleItems y extrae saleId + amount
sale_items = pd.read_csv(sale_items_path, usecols=['saleId', 'amount']).rename(columns={
    'saleId': 'sale_id',
    'amount': 'amount'
})

# Merge para añadir la columna 'amount' basado en sale_id
df = df.merge(sale_items, on='sale_id', how='left')

# Guarda el CSV final
output_csv  = 'client_util_sale_with_npp_and_amount.csv'
output_path = os.path.join(base_dir, output_csv)
df.to_csv(output_path, index=False)

print(f"CSV final con 'amount' añadido generado en: {output_path}")
