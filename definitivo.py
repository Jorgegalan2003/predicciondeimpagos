import os
import pandas as pd

# ————— Configura tu directorio de trabajo —————
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'

# ————— Nombres de los ficheros —————
sale_items_file = '_saleItems__202504221841.csv'
utilities_file  = 'utilities_202504222014.csv'
clients_file    = 'clients_202504221836.csv'
rates_file      = 'rates_202504222057.csv'

# ————— Rutas absolutas —————
sale_items_path = os.path.join(base_dir, sale_items_file)
utilities_path  = os.path.join(base_dir, utilities_file)
clients_path    = os.path.join(base_dir, clients_file)
rates_path      = os.path.join(base_dir, rates_file)

# ————— Verifica que existen los archivos —————
for p in (sale_items_path, utilities_path, clients_path, rates_path):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No encontrado: {p}")

# ————— Carga de las tablas, columnas necesarias —————
sale_items = pd.read_csv(
    sale_items_path,
    usecols=['saleId', 'utilityId', 'quantity', 'amount', 
             'cancellationDate', 'paymentDate']
)
utilities = pd.read_csv(
    utilities_path,
    usecols=['id', 'clientId', 'rateId']
)
clients = pd.read_csv(
    clients_path,
    usecols=['id', 'communityId', 'environmentId']
)
rates = pd.read_csv(
    rates_path,
    usecols=['id', 'nonPaymentPeriod']
)

# ————— Renombra columnas para mayor claridad —————
sale_items = sale_items.rename(columns={
    'saleId':           'sale_id',
    'utilityId':        'utility_id',
    'quantity':         'quantity',
    'amount':           'amount',
    'cancellationDate': 'cancellation_date',
    'paymentDate':      'payment_date'
})
utilities = utilities.rename(columns={
    'id':        'utility_id',
    'clientId':  'client_id',
    'rateId':    'rate_id'
})
clients = clients.rename(columns={
    'id':              'client_id',
    'communityId':     'community_id',
    'environmentId':   'environment_id'
})
rates = rates.rename(columns={
    'id':                 'rate_id',
    'nonPaymentPeriod':   'non_payment_period'
})

# ————— Parseo de fechas (mantener datetime) —————
sale_items['cancellation_date'] = pd.to_datetime(
    sale_items['cancellation_date'], errors='coerce', utc=True
)
sale_items['payment_date'] = pd.to_datetime(
    sale_items['payment_date'], errors='coerce', utc=True
)

# ————— Filtra sólo las ventas no canceladas —————
sale_items = sale_items[sale_items['cancellation_date'].isna()]

# ————— Realiza merges —————
df = (
    sale_items
    .merge(utilities, on='utility_id', how='left')
    .merge(rates,     on='rate_id',    how='left')
    .merge(clients,   on='client_id',  how='left')
)

# ————— Elimina filas con valores faltantes clave —————
df = df.dropna(subset=[
    'client_id', 'community_id', 'environment_id',
    'utility_id', 'sale_id', 'quantity',
    'rate_id', 'non_payment_period', 'amount',
    'payment_date'
])

# ————— Ordena por client_id y payment_date —————
df = df.sort_values(by=['client_id', 'payment_date'])

# ————— Selecciona columnas finales —————
final_df = df[[
    'client_id',
    'community_id',
    'environment_id',
    'utility_id',
    'sale_id',
    'quantity',
    'rate_id',
    'non_payment_period',
    'amount',
    'payment_date'
]]

# ————— Formatea payment_date a dd/mm/YYYY —————
final_df['payment_date'] = final_df['payment_date'].dt.strftime('%d/%m/%Y')

# ————— Guarda el CSV resultante —————
output_path = os.path.join(base_dir, 'client_util_sale_final.csv')
final_df.to_csv(output_path, index=False)

print(f"CSV final generado en: {output_path}")
