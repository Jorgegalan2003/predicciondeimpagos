import os
import pandas as pd

# Directorio donde están tus CSV
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'

# Nombres de los ficheros
final_csv_file  = 'client_util_sale_with_quantity.csv'
utilities_file  = 'utilities_202504222014.csv'

# Rutas absolutas
final_csv_path  = os.path.join(base_dir, final_csv_file)
utilities_path  = os.path.join(base_dir, utilities_file)

# Comprueba que existen los archivos
for path in (final_csv_path, utilities_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontrado: {path}")

# Carga del CSV generado previamente
final_df   = pd.read_csv(final_csv_path)

# Carga de utilities y extrae rateId asociado a cada utilityId
utilities = pd.read_csv(utilities_path, usecols=['id', 'rateId']).rename(columns={
    'id':     'utility_id',
    'rateId': 'rate_id'
})

# Merge para añadir rate_id a cada fila del CSV final, usando utility_id como clave
merged_df = final_df.merge(utilities, on='utility_id', how='left')

# Guarda el CSV enriquecido
output_file = 'client_util_sale_with_quantity_and_rate.csv'
output_path = os.path.join(base_dir, output_file)
merged_df.to_csv(output_path, index=False)

print(f"CSV con columna rate_id generado en: {output_path}")
