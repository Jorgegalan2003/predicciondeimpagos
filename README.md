# Predicción de impagos en proyectos de electrificación rural

Códigos empleados en el Trabajo Fin de Grado de **Jorge Galán** para la **ETSIT UPM**, cuyo objetivo es la predicción de impagos en proyectos de electrificación rural de la Fundación Acciona.

---

### 🔧 Preparación de datos

- [`clientes_y_ventas.py`](./clientes_y_ventas.py): Carga CSVs de clientes, servicios y ventas, los une y guarda el archivo consolidado `client_util_sale.csv`.
- [`clientes_y_ventas_ordenado.py`](./clientes_y_ventas_ordenado.py): Ordena el CSV anterior agrupando las ventas por cliente.

- Módulos que añaden variables:
  - [`rates.py`](./rates.py)
  - [`quantity.py`](./quantity.py)
  - [`nonPaymentPeriod.py`](./nonPaymentPeriod.py)
  - [`con_comunidad_y_país.py`](./con_comunidad_y_pa%C3%ADs.py)
  - [`amount.py`](./amount.py)
  - Todos estos están integrados en [`csv_completo.py`](./csv_completo.py), que ejecuta todas estas tareas a la vez.

- Limpieza y ordenación avanzada:
  - [`filtro.py`](./filtro.py): Elimina ventas con valores `NULL` en variables clave.
  - [`definitivo.py`](./definitivo.py): Realiza todas las tareas anteriores, más ordenamiento cronológico de ventas.

---

### 🌲 Algoritmos de Bosques Aleatorios

- [`bA_GLOBAL.py`](./bA_GLOBAL.py): Entrenamiento y evaluación global.
- [`bA_CLIENTES.py`](./bA_CLIENTES.py): Evaluación a nivel cliente.

---

### 🌎 División geográfica

- [`división_por_paises.py`](./divisi%C3%B3n_por_paises.py): Divide los datos en 5 datasets según el país.

---

### 🔁 Modelos LSTM

Entrenan y evalúan modelos LSTM para predecir el estado de pago de ventas:

- [`LSTM.py`](./LSTM.py)
- [`LSTM_POR_PESOS.py`](./LSTM_POR_PESOS.py)
- [`lstm_distribución_manual_de_pesos.py`](./lstm_distribuci%C3%B3n_manual_de_pesos.py)

---

### 🧠 Cálculo del estado de pago

- [`definitivo_con_estado_clientes.py`](./definitivo_con_estado_clientes.py): Calcula el estado ("CORRECTO", "IMPAGO"...).
- [`state_and_days.py`](./state_and_days.py): Añade días entre ventas.
- [`paymentPeriod.py`](./paymentPeriod.py): Añade la variable de periodo de pago.

---

### 📊 Escenarios de entrenamiento y evaluación (caso Perú)

- [`predict_autoregresive.py`](./predict_autoregresive.py)
- [`predict_autoregresive_pesos_manuales.py`](./predict_autoregresive_pesos_manuales.py)
- [`prioritize_recall_pesos_inversos.py`](./prioritize_recall_pesos_inversos.py)
- [`prioritize_recall_pesos_manueales.py`](./prioritize_recall_pesos_manueales.py)

---

### ⚙️ Técnicas avanzadas

- [`beam_search.py`](./beam_search.py)
- [`partial_teacher_forcing.py`](./partial_teacher_forcing.py)
- [`pesos_manuales_scheduled_sampling.py`](./pesos_manuales_scheduled_sampling.py)

---

### 🧮 Reglas y errores

- [`zero_rule.py`](./zero_rule.py): Algoritmo Zero Rule para benchmarking.
- [`ERRORES_RELATIVOS_CLIENTES.py`](./ERRORES_RELATIVOS_CLIENTES.py)
- [`ERRORES_RELATIVOS_CLIENTES_COMMUNITY.py`](./ERRORES_RELATIVOS_CLIENTES_COMMUNITY.py)

---

### ℹ️ Notas

- Todos los códigos que entrenan modelos pueden trabajar con datasets completos, divididos por país o filtrados.
- La preparación de datos es modular: se pueden añadir variables o dividir por países en cualquier momento según necesidad del usuario.

---

💡 Para más detalles, consulta el documento del TFG asociado.
