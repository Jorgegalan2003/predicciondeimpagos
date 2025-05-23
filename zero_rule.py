import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv("ventas_con_payment_period_CLIENTES_V3.csv")

# Asegurar que la columna de fecha esté en formato datetime
df['payment_date'] = pd.to_datetime(df['payment_date'])

# Inicializar listas para métricas y exportación
y_true = []
y_pred = []
export_rows = []

# Agrupar por cliente
for client_id, group in df.groupby('client_id'):
    group_sorted = group.sort_values(by='payment_date')

    if len(group_sorted) < 2:
        continue

    split_index = int(0.8 * len(group_sorted))
    train = group_sorted.iloc[:split_index]
    test = group_sorted.iloc[split_index:]

    mean_days = train['days_to_next_sale'].mean()

    for _, row in test.iterrows():
        true_val = row['days_to_next_sale']
        pred_val = mean_days
        abs_error = abs(true_val - pred_val)

        if 'rate_payment_period' not in row or pd.isna(row['rate_payment_period']):
            continue

        divisor = row['rate_payment_period'] * 3
        if divisor == 0:
            continue

        rel_error_pct = abs_error / divisor * 100

        y_true.append(true_val)
        y_pred.append(pred_val)

        export_rows.append({
            'client_id': client_id,
            'real_days_to_next_sale': true_val,
            'predicted_days_to_next_sale': pred_val,
            'error_absoluto': abs_error,
            'error_relativo_%': rel_error_pct
        })

# Calcular métricas globales
mae = mean_absolute_error(y_true, y_pred)
mean_divisor = (df['rate_payment_period'] * 3).mean()
mre = mae / mean_divisor

print(f"MAE (Error Absoluto Medio): {mae:.2f} días")
print(f"Error Relativo Medio (promediado sobre rate_payment_period*3): {mre:.4f} ({mre*100:.2f}%)")

# Guardar resultados
resultados_df = pd.DataFrame(export_rows)
resultados_df.to_csv("resultados_V3.csv", index=False)
print("Archivo 'resultados.csv' guardado correctamente.")

# ---------------- HISTOGRAMA CON TICKS CLAROS ----------------
plot_data = resultados_df["error_relativo_%"].copy()

# Crear bins con pasos de 1%
max_error = int(np.ceil(plot_data.max()))
bins = np.arange(0, max_error + 2, 1)  # +2 para incluir el extremo derecho

# Crear histograma
plt.figure(figsize=(14, 6))
plt.hist(plot_data, bins=bins, edgecolor="black")
plt.title("Distribución del Error Relativo HISTÓRICAS (%)")
plt.xlabel("Error relativo (%)")
plt.ylabel("Número de predicciones")
plt.grid(axis='y', linestyle="--", alpha=0.6)

# Mostrar ticks del eje X cada 10%
xtick_step = 10
xticks = np.arange(0, max_error + 1, xtick_step)
plt.xticks(xticks)

plt.tight_layout()
plt.savefig("error_histogram.png")
plt.show()
print("📷 Histograma guardado como: error_histogram.png")
