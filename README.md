Códigos empleados en el Trabajo Fin de Grado de Jorge Galán para la ETSIT UPM, cuyo objetivo es el de la predicción de impagos en proyectos de electrificación rural de la Fundación Acciona.

En primer lugar, está clientes_y_ventas.py. Este código carga tres archivos CSV que contienen información sobre ventas, servicios y clientes. Tras verificar que los archivos existen, extrae las columnas clave, las renombra y realiza uniones entre las tablas para relacionar cada venta con su servicio correspondiente y, a su vez, con el cliente asociado. Finalmente, reorganiza las columnas resultantes (client_id, utility_id, sale_id) y guarda esta información consolidada en un nuevo archivo CSV llamado client_util_sale.csv
Después, se creo el módulo clientes_y_ventas_ordenado.py, que lo que hace es ordenar el CSV generado con el módulo anterior y ordenarlo tanto por cliente, para que todas las ventas de cada cliente aparezcan de forma consecutiva, y por fecha, desde la venta más antigua a la más reciente.

Los módulos rates, quantity, nonPaymentPeriod, con_comunidad_y_país y amount añaden al CSV anterior las propiedades de ese mismo nombre al CSV. Toda esat funcionalidad se recoge en csv_completo.py, que hace simultáneamente todas las funcionalidades explicadas anteriormente.

Después, está el modulo filtro.py, que elimina todas aquellas ventas que tengan alguna de las variables independientes anteriormente mencionadas con valor NULL. El módulo definitivo.py realiza todas las funcionalidades explicadas hasta el momento, incluida este ultimo filtrado.

Por su parte, el módulo DATOS_REALES.py extrae de la base de datos los estados reales de las ventas pertenecientes al conjunto de ventas de validación, ya sea a nivel global o en el estudio por cliente, y añadirlos al CSV de datos, para luego poder entrenar y evaluar el modelo, sus métricas y precisión.
