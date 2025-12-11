# 📝 Evolución del Prompt para Sistema HAR

## 🔄 Iteración 1

### Solicitud Original

```
actua como un experto en prompt engineering y crea un buen prompt para esto: crea un modelo para HAR (human activity recognition) para el dataset MHealth (https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip) y guardalo, crea un backend y un frontend (front moderno, minimalista con estetica profesional listo para ser vendido). basate en las indicaciones de instrucciones.md
```

### ⚠️ Problema

Faltaba especificar el split de datos y la funcionalidad de carga de archivos.

---

## 🔄 Iteración 2

### Solicitud Original

```
me falto especificar que el modelo debe ser entrenado dejando fuera a 2 personas (.log) que luego seran usados en el front para probar el sistema (el front debe poder subir archivos .log).
```

### 📋 Respuesta del LLM

Eres un/a ML Engineer y Arquitecto/a Full-Stack Senior. Debes construir un proyecto completo, funcional de punta a punta y listo para producción y comercialización. El objetivo es crear un modelo de Human Activity Recognition usando el dataset MHealth y guardarlo, además de implementar un backend moderno para inferencia y un frontend moderno, minimalista y con estética profesional listo para ser vendido. Debes basarte de forma prioritaria y estricta en el archivo instrucciones.md si está disponible en el contexto. Si no está disponible o está incompleto, declara explícitamente los supuestos razonables que estás haciendo y continúa sin bloquearte.

Usa buenas prácticas de programación y mantén el código limpio, modular, testeable y fácil de mantener. Evita librerías o APIs deprecadas y prioriza versiones actuales estables. Para Python usa 3.11 o superior. Para backend usa FastAPI con Pydantic v2. Para frontend usa React con Vite y TypeScript. Incluye tooling moderno de calidad como Ruff, Black, MyPy y PyTest en Python y ESLint en el frontend. Optimiza cuando sea oportuno en rendimiento, mantenibilidad y experiencia de desarrollo.

El dataset MHealth debe descargarse automáticamente desde https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip, extraerse y parsearse. Implementa un pipeline de datos completo que incluya limpieza básica, segmentación en ventanas temporales con duración y solapamiento configurables, normalización o estandarización y una división clara en train, validación y test. Evita fugas de datos y prioriza el split por sujeto. Además, es obligatorio que el entrenamiento deje fuera a dos personas completas del dataset, identificadas por sus archivos .log, que no deben usarse ni directa ni indirectamente para entrenar, ajustar hiperparámetros o normalizadores. Esas dos personas deben quedar reservadas como un conjunto de prueba "demo" separado, pensado para validación final del sistema en interfaz. Debes documentar exactamente qué dos sujetos fueron excluidos y por qué regla se eligieron. Si instrucciones.md define cuáles son, respétalo; si no lo define, elige dos sujetos de forma determinística y explícita, y registra esa decisión en configuración para que sea reproducible. Fija semillas para reproducibilidad.

Entrena al menos un baseline clásico apropiado para series temporales transformadas a features y un modelo deep para señales temporales, como CNN1D, GRU o una combinación razonable. Reporta accuracy, macro F1 y matriz de confusión. Selecciona el mejor modelo según validación y guárdalo en un formato reproducible, usando joblib si es sklearn o torch.save si es PyTorch. Debes incluir scripts separados para entrenamiento, evaluación e inferencia standalone. Usa configuración mediante .env y/o un archivo config.yaml que incluya parámetros de ventana, solapamiento, lista de sujetos excluidos para demo, rutas de artefactos y versión del modelo.

El backend debe ser un servicio FastAPI que cargue el modelo entrenado al iniciar. Expón endpoints GET /health, GET /model-info y POST /predict. El endpoint de predicción debe aceptar tanto JSON con una o más ventanas de señales ya preprocesadas como también carga de archivos .log del formato MHealth para facilitar el flujo real. Si eliges soportar ambos modos, documenta claramente ambos contratos y provee utilidades de parsing y preprocesamiento idénticas a las usadas en entrenamiento, sin reutilizar estadísticas que incluyan a los sujetos reservados para demo. Valida entradas con Pydantic, maneja errores con mensajes claros y añade tests básicos del API. Provee documentación de ejecución local y con Docker si instrucciones.md lo permite o si no hay restricciones explícitas.

El frontend debe consumir el backend y ofrecer una experiencia de producto vendible. Implementa una interfaz moderna, minimalista y profesional con diseño consistente, buena tipografía y espaciado, componentes reutilizables y estados de UI bien resueltos. Debe existir una pantalla principal con explicación breve del producto, una sección para probar el sistema subiendo archivos .log, incluyendo específicamente los dos .log de las personas reservadas para demo, y un flujo alternativo para usar datos de ejemplo embebidos. Debe haber un botón para ejecutar predicción y una visualización elegante del resultado con etiqueta y confianza, además de manejo de loading, error y estado vacío. Incluye en el repositorio los dos archivos .log reservados para demo o, si por tamaño no es viable, incluye archivos recortados de ejemplo representativos y un mecanismo claro para ubicarlos localmente. El objetivo es que el usuario pueda probar el sistema end-to-end desde el frontend sin reentrenar.

Organiza todo como un monorepo con carpetas separadas para ml, backend y frontend, además de un README claro que explique cómo entrenar, evaluar, arrancar el API y levantar la UI. Añade configuración de calidad y formateo, y, si corresponde, un docker-compose para levantar backend y frontend fácilmente. Ignora datasets pesados en git y deja solo ejemplos pequeños versionados.

Entrega tu respuesta con un resumen breve de arquitectura, luego las decisiones técnicas justificadas, después un árbol de archivos y a continuación el contenido completo de los archivos principales necesarios para ejecutar el proyecto. Finaliza con instrucciones de ejecución paso a paso, una verificación de calidad que cubra tests, lint y build, y una sección de mejoras futuras opcionales. La solución es correcta si puedo entrenar el modelo con un comando, se genera y guarda el artefacto, el backend lo carga y predice, y el frontend permite subir .log y muestra resultados de forma elegante, usando como demo final los dos sujetos reservados que no participaron en el entrenamiento. Comienza ahora y respeta estrictamente instrucciones.md.

### ⚠️ Problema

Demasiado complejo: múltiples modelos, múltiples tipos de entrada.

---

## 🔄 Iteración 3

### Solicitud Original

```
no quiero varios modelos solo uno, que sea algo que procese los datos cree las ventanas y trabaje sobre esas (no necesito un modelo que trabaje directamente sobre series de tiempo). El front solo necesita poder aceptar los .log, no necesita otro tipo de entrada (igual para el backend), lo que si me gustaria es que haya una vista de informacion del modelo (por ejemplo una matriz de confusion sobre la precision del modelo si le subo un .log con la clase target)
```

### 📋 Respuesta del LLM

Eres un/a ML Engineer y Arquitecto/a Full-Stack Senior. Debes construir un proyecto completo, funcional de punta a punta y listo para producción y comercialización. El objetivo es crear un único modelo de Human Activity Recognition usando el dataset MHealth y guardarlo, además de implementar un backend moderno para inferencia y un frontend moderno, minimalista y con estética profesional listo para ser vendido. Debes basarte de forma prioritaria y estricta en el archivo instrucciones.md si está disponible en el contexto. Si no está disponible o está incompleto, declara explícitamente los supuestos razonables que estás haciendo y continúa sin bloquearte.

Usa buenas prácticas de programación y mantén el código limpio, modular, testeable y fácil de mantener. Evita librerías o APIs deprecadas y prioriza versiones actuales estables. Para Python usa 3.11 o superior. Para backend usa FastAPI con Pydantic v2. Para frontend usa React con Vite y TypeScript. Incluye tooling moderno de calidad como Ruff, Black, MyPy y PyTest en Python y ESLint en el frontend. Optimiza cuando sea oportuno en rendimiento, mantenibilidad y experiencia de desarrollo.

El dataset MHealth debe descargarse automáticamente desde https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip, extraerse y parsearse. Implementa un pipeline de datos completo que incluya limpieza básica, segmentación en ventanas temporales con duración y solapamiento configurables, extracción de características por ventana y normalización o estandarización. El modelo debe trabajar sobre esas ventanas y sus features derivadas; no necesitas un modelo que opere directamente sobre series de tiempo crudas. Usa un único algoritmo de clasificación adecuado para este enfoque y justifica la elección. La división en train, validación y test debe evitar fugas de datos y debe priorizar split por sujeto. Además, es obligatorio que el entrenamiento deje fuera a dos personas completas del dataset, identificadas por sus archivos .log, que no deben usarse ni directa ni indirectamente para entrenar, ajustar hiperparámetros ni calcular normalizadores. Esas dos personas deben quedar reservadas como un conjunto de prueba demo separado, pensado para validación final del sistema en interfaz. Debes documentar exactamente qué dos sujetos fueron excluidos y por qué regla se eligieron. Si instrucciones.md define cuáles son, respétalo; si no lo define, elige dos sujetos de forma determinística y explícita, persistiendo esa decisión en configuración para que sea reproducible. Fija semillas para reproducibilidad.

Entrena ese único modelo con el pipeline completo de ventanas y features y reporta métricas claras como accuracy, macro F1 y matriz de confusión sobre validación y test, dejando explícito el desempeño sobre el conjunto demo reservando a las dos personas excluidas. Guarda el artefacto del modelo en un formato reproducible junto con cualquier transformador necesario para que la inferencia sea idéntica al entrenamiento. Debes incluir scripts separados para entrenamiento, evaluación e inferencia standalone. Usa configuración mediante .env y/o un archivo config.yaml que incluya parámetros de ventana, solapamiento, lista de sujetos excluidos para demo, rutas de artefactos y versión del modelo.

El backend debe ser un servicio FastAPI que cargue el modelo entrenado y sus transformadores al iniciar. El backend solo necesita aceptar archivos .log del formato MHealth, no debe ofrecer entradas alternativas como JSON de ventanas. Expón endpoints GET /health, GET /model-info, POST /predict y un endpoint adicional para evaluación de un archivo etiquetado, por ejemplo POST /evaluate-log. El endpoint /predict debe recibir un .log, aplicar el mismo preprocesamiento, crear ventanas, extraer features, ejecutar inferencia y devolver una predicción por ventana y un resumen agregado por actividad con probabilidades o confianza. El endpoint /evaluate-log debe aceptar un .log que contenga la clase target y devolver métricas y una matriz de confusión calculada específicamente para ese archivo. Valida entradas, maneja errores con mensajes claros y añade tests básicos del API. Provee documentación de ejecución local y con Docker si instrucciones.md lo permite o si no hay restricciones explícitas.

El frontend debe consumir el backend y ofrecer una experiencia de producto vendible. Implementa una interfaz moderna, minimalista y profesional con diseño consistente, buena tipografía y espaciado y componentes reutilizables. El frontend solo necesita permitir subir archivos .log, tanto para predicción como para evaluación, sin otros métodos de entrada. Debe existir una vista principal para subir un .log y obtener predicciones y un resumen entendible. Debe existir una vista de información del modelo que permita subir un .log etiquetado y mostrar una matriz de confusión y métricas asociadas de forma elegante, además de mostrar también información del modelo cargada desde /model-info como versión, parámetros de ventana y métricas globales guardadas del entrenamiento. Incluye manejo de loading, error y estado vacío. Si es posible y compatible con las reglas de tamaño del repositorio, incluye los dos .log de las personas reservadas para demo para probar el flujo end-to-end desde el frontend sin reentrenar; si no es viable, incluye instrucciones claras de dónde colocarlos localmente.

Organiza todo como un monorepo con carpetas separadas para ml, backend y frontend, además de un README claro que explique cómo entrenar, evaluar, arrancar el API y levantar la UI. Añade configuración de calidad y formateo, y, si corresponde, un docker-compose para levantar backend y frontend fácilmente. Ignora datasets pesados en git y deja solo ejemplos pequeños versionados.

Entrega tu respuesta con un resumen breve de arquitectura, luego las decisiones técnicas justificadas, después un árbol de archivos y a continuación el contenido completo de los archivos principales necesarios para ejecutar el proyecto. Finaliza con instrucciones de ejecución paso a paso, una verificación de calidad que cubra tests, lint y build, y una sección de mejoras futuras opcionales. La solución es correcta si puedo entrenar el modelo con un comando, se genera y guarda el artefacto, el backend lo carga y predice recibiendo solo .log, el frontend permite subir .log y muestra resultados de forma elegante, y la vista de información del modelo puede calcular y mostrar una matriz de confusión al subir un .log etiquetado. Comienza ahora y respeta estrictamente instrucciones.md.

### Con esto se generó el siguiente contenido:

```
.
├── README.md
├── .env.example
├── .gitignore
├── config/config.yaml
├── docker-compose.yml
├── prompts/README.md
├── ml/
│ ├── requirements.txt
│ ├── train.py
│ ├── evaluate.py
│ ├── infer.py
│ ├── artifacts/
│ ├── demo_logs/README.md
│ └── src/mhealth/
│ ├── **init**.py
│ ├── config.py
│ ├── constants.py
│ ├── data.py
│ ├── preprocess.py
│ ├── modeling.py
│ ├── inference.py
│ └── utils.py
├── backend/
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── **init**.py
│ ├── app/
│ │ ├── **init**.py
│ │ ├── config.py
│ │ ├── main.py
│ │ ├── schemas.py
│ │ └── service.py
│ └── tests/test_api.py
└── frontend/
├── Dockerfile
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── .eslintrc.cjs
├── index.html
└── src/
├── App.tsx
├── api.ts
├── types.ts
├── main.tsx
└── index.css
```

### ⚠️ Problema

Interfaz poco atractiva y problemas de visualización.

## 🔄 Iteración 4

### Solicitud Original

```
cambia el front, la interfaz no me parece visual mente atractiva (incluso me parece basica) ademas de errores como que mostrar las ventanas en una lista no es comodo (seria bueno algo como una linea de tiempo), el texto del accuracy y macro F1 no se ve con el fondo, la matriz no cabe en el espacio asignado y se ve mal. Mejora y corrige la interfaz para que sea algo mas profesional, moderno y vendible
```

Con eso mas cambios manuales menores (como cambiar los nombres de las actividades) se llegó a una versión que era satisfactoria.

### ⚠️ Problema

El modelo seguía considerando la actividad 0, que representaba períodos de transición o inactividad, lo que afectaba negativamente su desempeño.

## 🔄 Iteración 5

### Solicitud Original

```
Corrige el modelo para que no considere la actividad 0 en el entrenamiento y evaluacion, tanto en el backend como en el frontend (el front debe dejar de mostrar la actividad 0 en las matrices de confusion y demas).
```

Con eso se mejora significativamente el desempeño del modelo.

Luego se realizaron cambios menores a la interfaz como cambiar textos, agregar hovers, mejorar la disposición de algunos elementos, etc. Con eso, se llegó a la versión final del proyecto.
