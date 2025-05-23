import pandas as pd

# Rutas de entrada
ventas_path = "ventas_unificadas.csv"
tarifas_path = "rates_202504222057.csv"
salida_path = "ventas_con_payment_period_CLIENTES_V3.csv"

# Cargar los archivos
ventas_df = pd.read_csv(ventas_path)
tarifas_df = pd.read_csv(tarifas_path)

# Merge para añadir el campo paymentPeriod
ventas_enriquecidas = ventas_df.merge(
    tarifas_df[['id', 'paymentPeriod']],
    left_on='rate_id',
    right_on='id',
    how='left'
)

# Renombrar la columna por claridad
ventas_enriquecidas.rename(columns={'paymentPeriod': 'rate_payment_period'}, inplace=True)

# Eliminar columna redundante del merge
ventas_enriquecidas.drop(columns=['id'], inplace=True)

# Guardar el nuevo CSV
ventas_enriquecidas.to_csv(salida_path, index=False)

print(f"Archivo generado: {salida_path}")
