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

# Comprueba que los ficheros existen
for path in (sale_items_path, utilities_path, clients_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontrado: {path}")

# Carga los CSV
sale_items = pd.read_csv(sale_items_path)
utilities  = pd.read_csv(utilities_path)
clients    = pd.read_csv(clients_path)

# Renombra y selecciona columnas clave, incluyendo 'quantity'
sale_items_df = sale_items[['saleId', 'utilityId', 'quantity']].rename(columns={
    'saleId':    'sale_id',
    'utilityId': 'utility_id',
    'quantity':  'quantity'
})

utilities_df = utilities[['id', 'clientId']].rename(columns={
    'id':        'utility_id',
    'clientId':  'client_id'
})

clients_df = clients[['id', 'communityId', 'environmentId']].rename(columns={
    'id':             'client_id',
    'communityId':    'community_id',
    'environmentId':  'environment_id'
})

# Merge para unir sale_items → utilities → clients
merged = (
    sale_items_df
    .merge(utilities_df, on='utility_id', how='left')
    .merge(clients_df, on='client_id', how='left')
)

# Selección y orden de columnas finales
final_df = merged[[
    'client_id',
    'community_id',
    'environment_id',
    'utility_id',
    'sale_id',
    'quantity'
]].sort_values(by='client_id')

# Guarda el CSV resultante
output_path = os.path.join(base_dir, 'client_util_sale_with_quantity.csv')
final_df.to_csv(output_path, index=False)

print(f"CSV con 'quantity' generado correctamente en: {output_path}")
