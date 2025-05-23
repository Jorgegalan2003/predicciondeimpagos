import os
import pandas as pd

# Directorio donde están tus CSV
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'

# Nombre del CSV que ya contiene rate_id
input_csv     = 'client_util_sale_with_quantity_and_rate.csv'
# Nombre del CSV de rates
rates_file    = 'rates_202504222057.csv'

# Rutas absolutas
input_path    = os.path.join(base_dir, input_csv)
rates_path    = os.path.join(base_dir, rates_file)

# Comprueba que los archivos existen
for path in (input_path, rates_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontrado: {path}")

# Carga del CSV con rate_id
df = pd.read_csv(input_path)

# Carga de rates y selecciona id + nonPaymentPeriod
rates = pd.read_csv(rates_path, usecols=['id', 'nonPaymentPeriod']).rename(columns={
    'id':               'rate_id',
    'nonPaymentPeriod': 'non_payment_period'
})

# Merge para añadir non_payment_period a cada fila, basado en rate_id
df = df.merge(rates, on='rate_id', how='left')

# Guarda el CSV resultante
output_csv = 'client_util_sale_with_npp.csv'
output_path = os.path.join(base_dir, output_csv)
df.to_csv(output_path, index=False)

print(f"CSV final con non_payment_period generado en: {output_path}")
