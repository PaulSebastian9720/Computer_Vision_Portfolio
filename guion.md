Claro pana, este queda más equilibrado: no tan largo, pero sí explica bien el proyecto. Para **2 personas** y video de **4 a 5 minutos**.

**Guion 2 Personas**
**0:00 - 0:30 | Presentación**

Mostrar: portada del proyecto, nombre, integrantes, una imagen/video de carretera.

**Persona 1:**

```text
Hello, my name is [Name 1].
```

**Persona 2:**

```text
And my name is [Name 2]. In this video, we present our computer vision project: a vehicle monitoring system focused on detecting cargo trucks in traffic scenes.
```

**Persona 1:**

```text
The project combines a classical detector developed in C++ with OpenCV, using HOG plus SVM, and a Telegram bot developed in Python that applies YOLO instance segmentation.
```

**0:30 - 1:20 | Problema y Objetivo**

Mostrar: videos de tráfico ecuatoriano, carretera, camiones.

**Persona 2:**

```text
The main problem is automatic truck detection in real road videos. In traffic scenes, trucks can appear from different angles, with different sizes, lighting conditions, and backgrounds.
```

**Persona 1:**

```text
Our target vehicle is the cargo truck. The objective is to detect this vehicle in a video or camera stream and automatically send visual evidence to the user through Telegram.
```

**Persona 2:**

```text
This type of system can be useful for road monitoring, vehicle counting, traffic analysis, and automatic alert generation.
```

**1:20 - 2:10 | Dataset y Entrenamiento**

Mostrar: carpetas de positivos/negativos, imágenes recortadas, ejemplos de camiones frontal/lateral, modelos `.yml`.

**Persona 1:**

```text
For the classical detector, we built a dataset with positive and negative samples. Positive images contain cargo trucks, while negative images contain roads, cars, vegetation, buildings, and other non-truck objects.
```

**Persona 2:**

```text
The dataset was balanced to avoid bias during training. This is important because if one class has many more images than the other, the SVM can learn incorrectly and produce many false positives or false negatives.
```

**Persona 1:**

```text
We trained HOG plus SVM models because HOG extracts shape and edge features, and SVM classifies whether a window belongs to the target vehicle or not.
```

**Persona 2:**

```text
We used two models: one for frontal truck views and another for lateral truck views. This helps the system detect trucks from different angles in the traffic videos.
```

**2:10 - 3:05 | Arquitectura del Sistema**

Mostrar: diagrama de arquitectura, carpetas `app`, `bot_telegram`, `models`, servidor corriendo.

**Persona 1:**

```text
The system architecture has two main components. The first one is the C++ desktop application. It reads a video file or a camera stream and runs the HOG plus SVM detector frame by frame.
```

**Persona 2:**

```text
When the C++ application detects the target vehicle, it saves the key frame, the HOG detection image, and a short five-second video clip.
```

**Persona 1:**

```text
Then, these files are sent to the Python server using a multipart API request. This implements the communication stream between the C++ application and the Telegram bot.
```

**Persona 2:**

```text
The second component is the Python bot. It receives the files, processes them with YOLO segmentation, and sends the results automatically to the registered Telegram user.
```

**3:05 - 4:10 | YOLO y Respuesta en Telegram**

Mostrar: Telegram con los tres mensajes: imagen HOG, imagen segmentada, clip segmentado.

**Persona 1:**

```text
For the deep learning part, we use a pretrained YOLO segmentation model. We do not retrain YOLO; we only use it in inference mode, as required in the project guide.
```

**Persona 2:**

```text
YOLO segments the objects it can recognize by default, such as trucks, cars, buses, motorcycles, people, traffic lights, and other traffic-related objects.
```

**Persona 1:**

```text
The Telegram bot sends three outputs. First, the HOG detection image with an alert message saying that the target vehicle was detected in the scene.
```

**Persona 2:**

```text
Second, it sends the same frame processed by YOLO, with colored segmentation masks and labels over the detected objects.
```

**Persona 1:**

```text
Third, it sends a short video clip where YOLO segmentation is applied frame by frame, so the user can observe the detected traffic scene over time.
```

**4:10 - 4:50 | Métricas y Resultados**

Mostrar: consola C++ con FPS/RAM/confianza, logs, detección funcionando.

**Persona 2:**

```text
The system also prints performance information in the console and logs. This includes FPS, RAM usage, confidence values, number of detections, and API communication information.
```

**Persona 1:**

```text
These metrics help us evaluate both the classical detector and the integrated system. They also show that the communication between C++ and Python is working correctly.
```

**Persona 2:**

```text
In our tests, the system was able to detect cargo trucks and send the required Telegram outputs automatically.
```

**4:50 - 5:10 | Conclusión**

Mostrar: pantalla final con Telegram o detección buena.

**Persona 1:**

```text
In conclusion, this project integrates classical computer vision and deep learning in a complete traffic monitoring pipeline.
```

**Persona 2:**

```text
The C++ application detects the target truck using HOG plus SVM, while the Python Telegram bot applies YOLO segmentation and sends visual evidence to the user.
```

**Ambos / Persona 1:**

```text
This completes the required flow: truck detection, API communication, Telegram alert, instance segmentation, video clip generation, and performance logging. Thank you.
```

Tip rápido: graben pantalla mientras muestran comandos, logs y Telegram. No se queden solo hablando, porque la rúbrica quiere ver arquitectura, dataset, resultados y funcionamiento real.