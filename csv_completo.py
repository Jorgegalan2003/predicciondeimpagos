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

# ————— Verifica existencia de todos los archivos —————
for p in (sale_items_path, utilities_path, clients_path, rates_path):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No encontrado: {p}")

# ————— Carga sólo las columnas necesarias —————
sale_items = pd.read_csv(sale_items_path, usecols=['saleId','utilityId','quantity','amount'])
utilities  = pd.read_csv(utilities_path,  usecols=['id','clientId','rateId'])
clients    = pd.read_csv(clients_path,    usecols=['id','communityId','environmentId'])
rates      = pd.read_csv(rates_path,      usecols=['id','nonPaymentPeriod'])

# ————— Renombrar columnas para claridad —————
sale_items_df = sale_items.rename(columns={
    'saleId':    'sale_id',
    'utilityId': 'utility_id',
    'quantity':  'quantity',
    'amount':    'amount'
})

utilities_df = utilities.rename(columns={
    'id':        'utility_id',
    'clientId':  'client_id',
    'rateId':    'rate_id'
})

clients_df = clients.rename(columns={
    'id':              'client_id',
    'communityId':     'community_id',
    'environmentId':   'environment_id'
})

rates_df = rates.rename(columns={
    'id':                 'rate_id',
    'nonPaymentPeriod':   'non_payment_period'
})

# ————— Encadenar merges en el orden correcto —————
final_df = (
    sale_items_df
    # 1) Asocia cada venta a su utility → obtiene client_id y rate_id
    .merge(utilities_df, on='utility_id', how='left')
    # 2) A través de rate_id añade non_payment_period
    .merge(rates_df,     on='rate_id',    how='left')
    # 3) Finalmente añade community_id y environment_id desde clients
    .merge(clients_df,   on='client_id',  how='left')
)

# ————— Selecciona y ordena columnas finales —————
final_df = final_df[[
    'client_id',
    'community_id',
    'environment_id',
    'utility_id',
    'sale_id',
    'quantity',
    'rate_id',
    'non_payment_period',
    'amount'
]].sort_values(by='client_id')

# ————— Guarda el CSV resultante —————
output_path = os.path.join(base_dir, 'client_util_sale_full.csv')
final_df.to_csv(output_path, index=False)

print(f"CSV final generado correctamente en: {output_path}")
