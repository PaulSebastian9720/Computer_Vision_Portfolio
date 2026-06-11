# III. Resultados — Calibración y Estimación de Profundidad

---

## 3.1 Parámetros obtenidos en la calibración estéreo

La calibración se realizó capturando 15 pares de imágenes de un tablero de ajedrez de 6×9 esquinas internas con cuadrados de 26 mm, usando el script `config/calibration.py`. Los parámetros resultantes se resumen a continuación.

### Matrices intrínsecas

| Parámetro | Cámara izquierda | Cámara derecha |
|---|---|---|
| Focal fx (px) | 232.88 | 226.40 |
| Focal fy (px) | 238.61 | 232.20 |
| Centro cx (px) | 157.18 | 171.12 |
| Centro cy (px) | 117.44 | 138.02 |

Las focales levemente diferentes entre ambas cámaras (232 vs 226 px) reflejan que las lentes del ESP32-CAM no son perfectamente idénticas. La diferencia es menor al 3%, lo cual es aceptable para esta clase de módulo de bajo costo.

### Vector de traslación T y baseline

El vector T = [81.85, 6.55, −22.43] mm describe la posición de la cámara derecha respecto a la izquierda. El baseline efectivo, calculado como `norm(T)`, es **85.12 mm**, con una separación física medida de 80 mm entre los módulos. Esto representa un **error de baseline del 6.4%**, originado principalmente en la componente diagonal de la traslación (eje Z: −22.43 mm), que indica que las cámaras no son exactamente coplanares. La componente vertical T[1] = 6.55 mm es pequeña (< 1 píxel en la imagen rectificada), lo que indica buena alineación vertical.

### Matriz de rotación R y ángulo de desviación

La matriz R obtenida presenta un ángulo de rotación de **10.5°** respecto a la identidad, calculado como:

```
θ = arccos((traza(R) − 1) / 2) = arccos((2.966 − 1) / 2) = 10.5°
```

Este ángulo indica que las cámaras tienen una leve convergencia óptica (no son perfectamente paralelas). Esta desviación es consecuencia de las limitaciones de montaje de los módulos ESP32-CAM, cuyas carcasas de plástico no garantizan una orientación precisa. Sin embargo, la función `cv::stereoRectify` compensa esta rotación aplicando transformaciones proyectivas a ambas imágenes, reduciendo el error epipolar residual a menos de 2 píxeles después de la rectificación.

El error epipolar antes de la rectificación puede estimarse como:

```
Desplazamiento vertical máximo ≈ altura_imagen × sin(10.5°) ≈ 240 × 0.182 ≈ 43.6 px
Después de rectificación: < 2 px
```

---

## 3.2 Análisis del error en la estimación de profundidad Z

### Error sistemático por baseline

El error de baseline del 6.4% se traduce directamente en una sobreestimación sistemática de Z en el mismo porcentaje. Si la distancia real es 50 cm, el sistema reporta aproximadamente 53.2 cm sin corrección.

| Distancia real | Z reportado (sin corrección) | Error sistemático |
|---|---|---|
| 20 cm | 21.3 cm | +6.4% |
| 30 cm | 31.9 cm | +6.4% |
| 50 cm | 53.2 cm | +6.4% |
| 70 cm | 74.5 cm | +6.4% |
| 100 cm | 106.4 cm | +6.4% |

Este error es **corregible** mediante la calibración manual implementada en el sistema (tecla C): el usuario posiciona un objeto a distancia conocida, ingresa el valor real, y el sistema calcula un factor de escala que elimina el error sistemático para todas las distancias.

### Error aleatorio por ruido de disparidad

Independientemente del error de calibración, la disparidad medida por StereoSGBM tiene un ruido de ±1–2 píxeles frame a frame. Este ruido produce un error aleatorio en Z que depende de la distancia, ya que la relación Z = f·B/d es no lineal:

| Distancia | Disparidad teórica | Error ±2 px en disparidad | Error en Z |
|---|---|---|---|
| 20 cm | 99 px | ±2% | ±0.4 cm |
| 30 cm | 66 px | ±3% | ±0.9 cm |
| 50 cm | 40 px | ±5% | ±2.5 cm |
| 70 cm | 28 px | ±7% | ±4.9 cm |
| 100 cm | 20 px | ±10% | ±10.0 cm |

El error aumenta cuadráticamente con la distancia porque la disparidad disminuye. Para el rango de trabajo del efecto AR (30–130 cm), el error aleatorio oscila entre ±1 y ±15 cm. El filtro de Kalman reduce este error al combinar múltiples mediciones consecutivas, obteniendo en la práctica errores de ±0.5 a ±5 cm en el rango de interés.

### Resultado con calibración manual aplicada

Aplicando la corrección de escala (tecla C), el error sistemático se cancela. Los errores residuales corresponden únicamente al ruido de disparidad filtrado por Kalman:

| Distancia | Error Z (post-calibración) |
|---|---|
| 30 cm | ±0.6 cm |
| 50 cm | ±1.5 cm |
| 70 cm | ±3.2 cm |
| 100 cm | ±6.0 cm |

Estos valores son suficientes para el propósito del sistema: el efecto AR opera con rangos de Z de 10 cm de amplitud por zona ([30–50], [50–80], [80–130] cm), por lo que errores menores a 6 cm no afectan la clasificación de intensidad del efecto.

---

## 3.3 Estabilidad temporal del mapa de disparidad

El mapa de disparidad sin filtrado presenta variaciones frame a frame de ±15–20% en zonas de baja textura (superficies homogéneas, ropa). Se implementaron dos mecanismos de estabilización:

**Ring buffer (16 muestras):** guarda las últimas 16 mediciones de Z en el parche central y aplica la mediana. Reduce los outliers individuales causados por reflejos o cambios repentinos de iluminación.

**Filtro de Kalman 1D (Q=10, R=300):** la relación R/Q=30 indica alta desconfianza en cada medición individual y alta confianza en la predicción del modelo. Esto produce una estimación suave que converge en 5–8 frames hacia el valor real sin oscilar.

**Suavizado temporal del heatmap (α=0.4):** el mapa de colores mostrado en pantalla se calcula como un promedio ponderado (40% frame nuevo, 60% acumulado) para reducir el parpadeo visual sin introducir retraso perceptible.

---

## 3.4 Limitaciones identificadas

1. **Ángulo de convergencia:** Los 10.5° de rotación entre cámaras es la principal fuente de imprecisión. Se origina en el montaje físico de los módulos ESP32-CAM y no puede eliminarse completamente por software. Una solución de hardware (soporte impreso en 3D con guías de alineación) reduciría este ángulo a < 2°.

2. **Coeficientes de distorsión elevados:** El módulo izquierdo tiene k2=−0.92 y k3=1.63, valores altos que indican distorsión de barril significativa. La corrección de distorsión de alto orden es más sensible al ruido en las esquinas del tablero durante la calibración.

3. **Resolución limitada (320×240):** Con imágenes pequeñas, cada píxel de disparidad equivale a una variación de Z mayor, amplificando el error aleatorio a distancias largas.
