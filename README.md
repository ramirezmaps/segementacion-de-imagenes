# 🌳 Segmentación de Árboles en Primer Plano (Machine Learning + Streamlit Web App)

**Autor y Desarrollo**: **Nacho**

Aplicación Web interactiva desarrollada con **Streamlit** y **Python** que utiliza Inteligencia Artificial y Visión por Computadora para generar la máscara de segmentación de un **árbol en primer plano**, filtrando y excluyendo los árboles o vegetación situados en el **segundo plano (fondo)** y descartando automáticamente el **suelo/terreno**.

---

## 👤 Autor

Desarrollado y creado por **Nacho**.

---

## 🌟 Características Principales

- **Estimación de Profundidad Relativa (Monocular Depth Estimation)**: Utiliza redes neuronales (DPT / MiDaS) para predecir la distancia de cada objeto respecto a la cámara.
- **Detección Estructural Saliente (U2-Net)**: Combina la información de profundidad con contornos morfológicos de alta precisión para aislar la copa, tronco y ramas del árbol principal.
- **Exclusión Automática de Suelo/Terreno**: Algoritmo especializado que filtra la hierba, asfalto, pasto o tierra de la base.
- **Control Interactivo de Planos (Depth Threshold Slider)**: Permite ajustar en tiempo real el umbral de distancia entre el primer plano y el fondo.
- **Visualizador 4-Planos**:
  1. Imagen Original
  2. Mapa de Profundidad (Heatmap Inferno)
  3. Máscara Binaria Monocromática (Blanco = Primer Plano, Sin Suelo)
  4. Recorte Aislado con Fondo Transparente (PNG RGBA)
- **Carga Flexible de Imágenes**: Soporta JPG, PNG, WEBP, TIFF y cuenta con **imágenes de prueba sintéticas** para demostración rápida sin subir archivos.
- **Exportación en Alta Calidad**: Botones para descargar la máscara binaria y el recorte del árbol aislado.

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Requisitos Previos
Tener Python 3.10, 3.11 o 3.12 instalado.

### 2. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación Web
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

---

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework de interfaz de usuario web.
- **PyTorch & Hugging Face Transformers**: Modelo DPT-Hybrid MiDaS para estimación de profundidad.
- **rembg & U2-Net**: Extracción de máscaras salientes de árboles.
- **scikit-image & NumPy**: Procesamiento avanzado de imágenes, análisis de regiones y morfología de máscaras.
