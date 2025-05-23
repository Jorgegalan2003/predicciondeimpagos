import os
import pandas as pd

# Directorio donde están tus CSV
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'

# Nombres exactos de los ficheros
sale_items_file = '_saleItems__202504221841.csv'
utilities_file  = 'utilities_202504222014.csv'
clients_file    = 'clients_202504221836.csv'

# Rutas absolutas
sale_items_path = os.path.join(base_dir, sale_items_file)
utilities_path  = os.path.join(base_dir, utilities_file)
clients_path    = os.path.join(base_dir, clients_file)

# Comprueba que los ficheros existen (opcional, para depurar)
for path in (sale_items_path, utilities_path, clients_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontrado: {path}")

# Carga los CSV
sale_items = pd.read_csv(sale_items_path)
utilities  = pd.read_csv(utilities_path)
clients    = pd.read_csv(clients_path)

# Renombra y selecciona columnas clave
sale_items_df = sale_items[['saleId', 'utilityId']].rename(columns={
    'saleId':    'sale_id',
    'utilityId': 'utility_id',
})
utilities_df = utilities[['id', 'clientId']].rename(columns={
    'id':        'utility_id',
    'clientId':  'client_id',
})
clients_df = clients[['id']].rename(columns={
    'id': 'client_id',
})

# Merge para unir las tres tablas
merged = (
    sale_items_df
    .merge(utilities_df, on='utility_id')
    .merge(clients_df, on='client_id')
)

# Coloca las columnas en el orden deseado
final_df = merged[['client_id', 'utility_id', 'sale_id']]

# Guarda el CSV resultante
output_path = os.path.join(base_dir, 'client_util_sale.csv')
final_df.to_csv(output_path, index=False)

print(f"CSV generado correctamente en: {output_path}")
