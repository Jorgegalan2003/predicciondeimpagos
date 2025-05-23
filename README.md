# Predicción de impagos en proyectos de electrificación rural

Códigos empleados en el Trabajo Fin de Grado de **Jorge Galán** para la **ETSIT UPM**, cuyo objetivo es la predicción de impagos en proyectos de electrificación rural de la Fundación Acciona.

---

### 🔧 Preparación de datos

- [`clientes_y_ventas.py`](./clientes_y_ventas.py): Carga tres archivos CSV con información sobre ventas, servicios y clientes. Verifica su existencia, extrae y renombra columnas clave, une las tablas para relacionar cada venta con su servicio y cliente, reorganiza las columnas resultantes (`client_id`, `utility_id`, `sale_id`) y guarda el resultado en `client_util_sale.csv`.

- [`clientes_y_ventas_ordenado.py`](./clientes_y_ventas_ordenado.py): Ordena el CSV generado por el módulo anterior para que todas las ventas de cada cliente aparezcan agrupadas de forma consecutiva.

- Módulos que añaden variables específicas al CSV:
  - [`rates.py`](./rates.py): Añade la tarifa de cada venta.
  - [`quantity.py`](./quantity.py): Añade la cantidad de consumo.
  - [`nonPaymentPeriod.py`](./nonPaymentPeriod.py): Añade el periodo de impago.
  - [`con_comunidad_y_país.py`](./con_comunidad_y_país.py): Añade la comunidad y país del cliente.
  - [`amount.py`](./amount.py): Añade el importe de cada venta.
  - Todos ellos se integran en [`csv_completo.py`](./csv_completo.py), que realiza simultáneamente todas las operaciones anteriores sobre el CSV.

- Limpieza y organización avanzada:
  - [`filtro.py`](./filtro.py): Elimina todas las ventas con alguna de las variables independientes anteriores con valor `NULL`.
  - [`definitivo.py`](./definitivo.py): Ejecuta todas las funcionalidades anteriores, incluyendo el filtrado y la ordenación cronológica de las ventas por cliente, de la más antigua a la más reciente.

---

### 🌲 Algoritmos de Bosques Aleatorios

- [`bA_GLOBAL.py`](./bA_GLOBAL.py): Evalúa las métricas de clasificación, reporte y matriz de confusión a nivel global para todos los clientes usando Random Forest.
- [`bA_CLIENTES.py`](./bA_CLIENTES.py): Aplica las métricas anteriores a nivel individual por cliente. Los resultados obtenidos con estos módulos ayudaron a descartar el uso de Random Forest para este caso de estudio.

---

### 🌎 División geográfica

- [`división_por_paises.py`](./división_por_paises.py): Divide el CSV completo en cinco datasets separados por país, basándose en el identificador de país, permitiendo análisis específicos por región.

---

### 🔁 Modelos LSTM

Estos módulos entrenan modelos LSTM para predecir el estado de pago de clientes basado en datos de ventas y consumo:

- [`LSTM.py`](./LSTM.py): Modelo básico con pesos por defecto para clases.
- [`LSTM_POR_PESOS.py`](./LSTM_POR_PESOS.py): Asigna pesos inversamente proporcionales a la frecuencia de las clases para compensar desbalance.
- [`lstm_distribución_manual_de_pesos.py`](./lstm_distribución_manual_de_pesos.py): Asigna pesos manuales, enfatizando la detección de impagos.

Los tres módulos cargan datos, normalizan entradas, dividen en entrenamiento/validación, y entrenan modelos con PyTorch. Se observó que pesos manuales e inversos ofrecían resultados similares.

---

### 🧠 Cálculo del estado de pago

- [`definitivo_con_estado_clientes.py`](./definitivo_con_estado_clientes.py): Calcula el estado final de pago de cada venta y lo registra en la columna `state`.
- [`state_and_days.py`](./state_and_days.py): Añade los días reales transcurridos entre ventas de un cliente.
- [`paymentPeriod.py`](./paymentPeriod.py): Añade la variable `paymentPeriod` basada en la tarifa del cliente.

---

### 📊 Escenarios de entrenamiento y evaluación (caso Perú)

Se probaron cuatro escenarios sobre datos de Perú, el país con clases más desbalanceadas:

- [`predict_autoregresive.py`](./predict_autoregresive.py): Usa pesos inversos y prioriza la validación (`Validation Loss`).
- [`predict_autoregresive_pesos_manuales.py`](./predict_autoregresive_pesos_manuales.py): Igual al anterior, pero con pesos manuales.
- [`prioritize_recall_pesos_inversos.py`](./prioritize_recall_pesos_inversos.py): Enfatiza el `Recall` de la clase impago usando pesos inversos.
- [`prioritize_recall_pesos_manueales.py`](./prioritize_recall_pesos_manueales.py): Igual, pero con pesos manuales.

Todos entrenan y predicen con LSTM, generando reportes de clasificación para comparar métodos.

---

### ⚙️ Técnicas avanzadas

- [`beam_search.py`](./beam_search.py): Entrena y evalúa modelos LSTM con la técnica de búsqueda Beam Search.
- [`partial_teacher_forcing.py`](./partial_teacher_forcing.py): Aplica Partial Teacher Forcing durante el entrenamiento.
- [`pesos_manuales_scheduled_sampling.py`](./pesos_manuales_scheduled_sampling.py): Implementa Scheduled Sampling con pesos manuales, técnica que mostró mejor eficiencia, pero menor rendimiento general que las anteriores.

---

### 🧮 Reglas y errores

- [`zero_rule.py`](./zero_rule.py): Implementa el algoritmo Zero Rule para predecir la media de días entre ventas. Evalúa el rendimiento con errores absoluto y relativo.
- [`ERRORES_RELATIVOS_CLIENTES.py`](./ERRORES_RELATIVOS_CLIENTES.py): Calcula el error relativo por cliente para el análisis posterior de histogramas.
- [`ERRORES_RELATIVOS_CLIENTES_COMMUNITY.py`](./ERRORES_RELATIVOS_CLIENTES_COMMUNITY.py): Similar al anterior, pero incorporando la variable `community_id` para evaluar influencia geográfica.

---

### ℹ️ Notas

- Todos los códigos de modelado pueden trabajar con datasets globales, divididos por país, o filtrados.
- Los scripts de preparación son modulares y pueden ejecutarse en cualquier orden. La división por país y la incorporación de variables son personalizables.

---

💡 Para más información técnica y fundamentos, consulta el documento completo del TFG.
