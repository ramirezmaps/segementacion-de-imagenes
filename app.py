"""
Aplicación Web Streamlit - Segmentación de Árboles por Planos de Profundidad (Machine Learning)
Permite subir imágenes de árboles y generar la máscara del árbol en primer plano aislándolo del segundo plano y del suelo.

Autor: Nacho
"""

import streamlit as st
import numpy as np
from PIL import Image

from depth_estimator import estimate_depth
from tree_segmenter import segment_foreground_tree, get_salient_mask
from utils import create_overlay, extract_cutout, image_to_bytes, create_synthetic_sample_trees

__author__ = "Nacho"

@st.cache_data(show_spinner=False)
def cached_estimate_depth(_img: Image.Image, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    return estimate_depth(_img, model_name=model_name)

@st.cache_data(show_spinner=False)
def cached_get_salient_mask(_img: Image.Image):
    return get_salient_mask(_img)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Segmentación de Árboles por Profundidad | Desarrollado por Nacho",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar el diseño de la UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E4620;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 0.5rem;
    }
    .author-badge {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid #86EFAC;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #15803D;
        color: white;
        font-weight: 600;
    }
    .stDownloadButton button:hover {
        background-color: #166534;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Encabezado con atribución de autor
    st.markdown('<div class="main-title">🌳 Segmentación de Árboles en Primer Plano</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-badge">👨‍💻 Desarrollado por Nacho</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Aplica Machine Learning (Estimación de Profundidad Monocular DPT + Segmentación Saliente U2-Net) '
        'para aislar el árbol en <b>primer plano</b> y excluir la vegetación de <b>segundo plano</b> y el <b>suelo/terreno</b>.</div>',
        unsafe_allow_html=True
    )
    
    # Menú Lateral (Sidebar)
    st.sidebar.markdown("### 👨‍💻 Autor: **Nacho**")
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Carga de Imagen")
    source_option = st.sidebar.radio(
        "Seleccionar origen de la imagen:",
        ["Subir mi propia imagen", "Usar imagen de prueba sintética"]
    )
    
    image_input = None
    
    if source_option == "Subir mi propia imagen":
        uploaded_file = st.sidebar.file_uploader(
            "Cargar foto de un árbol:",
            type=["jpg", "jpeg", "png", "webp", "tiff"]
        )
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
    else:
        sample_dict = create_synthetic_sample_trees()
        sample_name = st.sidebar.selectbox("Selecciona una muestra:", list(sample_dict.keys()))
        image_input = sample_dict[sample_name]

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Ajuste de Planos y Profundidad")
    
    depth_model_choice = st.sidebar.selectbox(
        "Modelo de Profundidad IA:",
        ["Depth Anything V2 Small (Ultra Rápido)", "DPT-Hybrid MiDaS (Clásico)"],
        index=0,
        help="Depth Anything V2 ofrece mayor precisión de bordes y una velocidad >2x superior en CPU."
    )
    model_name = "depth-anything/Depth-Anything-V2-Small-hf" if "Depth Anything" in depth_model_choice else "intel/dpt-hybrid-midas"

    depth_threshold = st.sidebar.slider(
        "Corte de Profundidad (Primer Plano vs Segundo Plano):",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.02,
        help="Valores más altos aíslan únicamente los objetos más cercanos a la cámara. Valores más bajos incluyen planos más lejanos."
    )
    
    saliency_weight = st.sidebar.slider(
        "Peso de Segmentación de Objetos (Salience U2-Net):",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Combina el mapa de profundidad con la forma estructural del árbol detectada por saliencia."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🚫 Filtro de Suelo / Terreno")
    
    filter_ground = st.sidebar.checkbox(
        "Excluir Suelo y Terreno de la Máscara",
        value=True,
        help="Detecta y elimina automáticamente las zonas llanas de hierba, pasto, tierra o asfalto en la base."
    )
    
    ground_sensitivity = st.sidebar.slider(
        "Sensibilidad de Exclusión de Suelo:",
        min_value=0.1,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Aumenta la sensibilidad si parte del suelo sigue apareciendo en el corte inferior."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🧹 Filtros Avanzados y Bordes")
    
    morph_kernel = st.sidebar.slider(
        "Suavizado Morfológico de Bordes (px):",
        min_value=1,
        max_value=15,
        value=5,
        step=2,
        help="Elimina ruido de píxeles aislados y rellena pequeñas brechas en las hojas."
    )
    
    min_area_filter = st.sidebar.checkbox(
        "Filtrar componentes pequeños del fondo",
        value=True,
        help="Elimina pequeñas manchas de vegetación lejana que superen el umbral de profundidad."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Opciones de Visualización")
    
    color_choice = st.sidebar.selectbox(
        "Color de la Máscara de Superposición:",
        ["Verde Esmeralda", "Azul Cian", "Rojo Carmín", "Amarillo Neón"]
    )
    color_map = {
        "Verde Esmeralda": (0, 255, 128),
        "Azul Cian": (0, 200, 255),
        "Rojo Carmín": (255, 50, 80),
        "Amarillo Neón": (255, 235, 20)
    }
    overlay_color = color_map[color_choice]
    
    overlay_alpha = st.sidebar.slider(
        "Opacidad de Superposición:",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05
    )

    # Procesamiento principal si hay imagen cargada
    if image_input is not None:
        st.subheader("📸 Resultados del Análisis de Planos y Filtrado de Suelo")
        with st.spinner("Ejecutando modelo de estimación de profundidad y filtrado de terreno..."):
            # 1. Estimación de Profundidad y Saliencia con Caché en Memoria (Instantáneo al mover sliders)
            depth_norm, depth_colormap = cached_estimate_depth(image_input, model_name=model_name)
            salient_alpha = cached_get_salient_mask(image_input)
            
            # 2. Segmentación del Árbol en Primer Plano con Puerta Estricta de Profundidad Física
            binary_mask, stats = segment_foreground_tree(
                image=image_input,
                depth_norm=depth_norm,
                depth_threshold=depth_threshold,
                saliency_weight=saliency_weight,
                morph_kernel_size=morph_kernel,
                min_area_filter=min_area_filter,
                filter_ground=filter_ground,
                ground_sensitivity=ground_sensitivity,
                salient_alpha=salient_alpha
            )
            
            # 3. Generación de Recorte y Overlay
            mask_pil = Image.fromarray(binary_mask)
            overlay_img = create_overlay(image_input, binary_mask, color_rgb=overlay_color, alpha=overlay_alpha)
            cutout_img = extract_cutout(image_input, binary_mask)

        # Mostrar métricas resumidas
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Resolución de la Imagen", f"{stats['width']} x {stats['height']} px")
        with col_m2:
            st.metric("Cobertura Árbol Primer Plano", f"{stats['fg_percentage']:.2f}%")
        with col_m3:
            st.metric("Umbral de Plano Seleccionado", f"{stats['depth_threshold_used']:.2f}")
        with col_m4:
            st.metric("Exclusión de Suelo", "Activa 🚫" if filter_ground else "Inactiva")

        st.markdown("<br>", unsafe_allow_html=True)

        # Disposición en 4 Columnas para comparar los planos
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 1. Imagen Original")
            st.image(image_input, use_container_width=True)
            
        with col2:
            st.markdown("### 2. Mapa de Profundidad")
            st.image(depth_colormap, use_container_width=True)
            st.caption("🔥 **Cálido/Amarillo**: Primer Plano (Cercano) | ❄️ **Oscuro**: Segundo Plano (Lejano)")
            
        with col3:
            st.markdown("### 3. Máscara Árbol (Sin Suelo)")
            st.image(mask_pil, use_container_width=True)
            st.caption("⬜ **Blanco**: Árbol Primer Plano | ⬛ **Negro**: Suelo/Fondo/Segundo Plano")
            
        with col4:
            st.markdown("### 4. Recorte Transparente")
            st.image(cutout_img, use_container_width=True)
            st.caption("✨ Árbol aislado (sin suelo ni fondo)")

        st.markdown("---")
        
        # Comparación Interactiva y Descargas
        st.subheader("📥 Exportación y Descargas")
        
        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        
        with col_d1:
            st.markdown("#### Superposición Ajustada")
            st.image(overlay_img, use_container_width=True)
            
        with col_d2:
            st.markdown("#### Descargar Máscara Binaria")
            mask_bytes = image_to_bytes(mask_pil, format="PNG")
            st.download_button(
                label="⬇️ Descargar Máscara (PNG 8-bit)",
                data=mask_bytes,
                file_name="mascara_arbol_sin_suelo.png",
                mime="image/png"
            )
            st.info("Formato PNG monocromático con el suelo y fondo filtrados.")
            
        with col_d3:
            st.markdown("#### Descargar Árbol Aislado")
            cutout_bytes = image_to_bytes(cutout_img, format="PNG")
            st.download_button(
                label="⬇️ Descargar Recorte (PNG RGBA)",
                data=cutout_bytes,
                file_name="arbol_primer_plano_sin_suelo.png",
                mime="image/png"
            )
            st.info("Imagen PNG con canal alfa transparente lista para composiciones.")

    else:
        st.info("👆 Por favor, sube una imagen de un árbol desde el panel izquierdo o selecciona una de las muestras sintéticas para comenzar.")

    # Pie de página con créditos de autoría
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #718096; font-size: 0.95rem; font-weight: 500;'>"
        "Desarrollado y Creado por <b>Nacho</b> | Aplicación de Machine Learning para Segmentación de Árboles en Primer Plano"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
