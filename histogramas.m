% Cargar datos desde el archivo CSV
filename = 'errores_relativos_Filipinas_caso4.csv';
data = readtable(filename);

% Escalar errores relativos a porcentaje
relative_error_percent = data.relative_error * 100;

% Limitar el valor máximo al último bin (100–105%)
relative_error_percent_clipped = min(relative_error_percent, 104.999);

% Definir bordes de bins: 0% a 105%, en pasos de 5%
bin_edges = 0:5:105;

% Crear histograma con conteo absoluto
figure;
histogram(relative_error_percent_clipped, bin_edges, 'Normalization', 'count');

% Configurar el gráfico
xlabel('Error relativo (%)');
ylabel('Cantidad de observaciones');
title('Histograma del Error Relativo (%) para las ventas de Filipinas de Proteo V3 con la comunidad');

% Etiquetas del eje X centradas en cada barra
centers = bin_edges(1:end-1) + 2.5;
xticks(centers);
xticklabels([arrayfun(@(x) sprintf('%d%%', x), 0:5:95, 'UniformOutput', false), '>100%']);

grid on;
