import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# ——— Configuración ———
base_dir = r'C:\Users\34649\Desktop\TELECO\4º\TFG\DATOS'
real_csv = os.path.join(base_dir, 'client_util_sale_with_state.csv')       # datos reales
preds_csv = os.path.join(base_dir, 'predicciones_CLIENTES.csv')           # predicciones modelo
output_csv = os.path.join(base_dir, 'metricas_global_y_paises.csv')

# ——— Cargar archivos ———
df_real = pd.read_csv(real_csv, usecols=['sale_id', 'state', 'environment_id'])
df_preds = pd.read_csv(preds_csv, usecols=['sale_id', 'pred_state'])

# ——— Unir por sale_id ———
df = pd.merge(df_real, df_preds, on='sale_id', how='inner')

# ——— Convertir 'state' a numérico si es texto ———
if df['state'].dtype == 'object':
    estado_map = {'CORRECTO': 0, 'PENDIENTE_DE_PAGO': 1, 'IMPAGO': 2}
    df['state'] = df['state'].map(estado_map)

# ——— Diccionario de environment_id a país ———
env_map = {
    1: 'Peru',
    2: 'Mexico',
    3: 'Panama',
    4: 'Chile',
    5: 'España',
    6: 'Filipinas',
    7: 'Desconocido',
    8: 'Republica_Dominicana'
}

# ——— Función para calcular métricas ———
def calcular_metricas(df_sub, scope='GLOBAL'):
    y_true = df_sub['state']
    y_pred = df_sub['pred_state']
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)

    labels_presentes = sorted(pd.unique(y_true))
    nombre_clases = ['CORRECTO', 'PENDIENTE_DE_PAGO', 'IMPAGO']
    nombre_clases_presentes = [nombre_clases[i] for i in labels_presentes]

    report = classification_report(
        y_true, y_pred,
        labels=labels_presentes,
        target_names=nombre_clases_presentes,
        output_dict=True
    )

    filas = []
    for clase in nombre_clases_presentes:
        r = report[clase]
        filas.append({
            'ambito': scope,
            'estado': clase,
            'precision_%': round(r['precision'] * 100, 2),
            'recall_%':    round(r['recall'] * 100, 2),
            'f1_score_%':  round(r['f1-score'] * 100, 2),
            'soporte':     int(r['support'])
        })

    for avg in ['macro avg', 'weighted avg']:
        r = report[avg]
        filas.append({
            'ambito': scope,
            'estado': avg.replace(' avg', '_avg'),
            'precision_%': round(r['precision'] * 100, 2),
            'recall_%':    round(r['recall'] * 100, 2),
            'f1_score_%':  round(r['f1-score'] * 100, 2),
            'soporte':     int(r['support'])
        })

    filas.append({
        'ambito': scope,
        'estado': 'TOTAL_ACCURACY',
        'precision_%': acc,
        'recall_%': '',
        'f1_score_%': '',
        'soporte': ''
    })
    return filas

# ——— Métricas globales ———
metricas_totales = calcular_metricas(df)

# ——— Métricas por país ———
for env_id, pais in env_map.items():
    df_sub = df[df['environment_id'] == env_id]
    if not df_sub.empty:
        metricas_totales.extend(calcular_metricas(df_sub, scope=pais))

# ——— Exportar resultados ———
df_out = pd.DataFrame(metricas_totales)
df_out.to_csv(output_csv, index=False)
print(f"✅ Métricas globales y por país exportadas en: {output_csv}")
