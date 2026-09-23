"""
Módulo de Estimación de Profundidad (Depth Estimation)
Utiliza modelos de visión por computadora (DPT/MiDaS de Hugging Face) para predecir la profundidad relativa
de cada píxel en la imagen. Usa Matplotlib colormaps para visualizaciones sin dependencias DLL.
"""

import numpy as np
import matplotlib.cm as cm
import torch
from PIL import Image
from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_depth_pipeline(model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    """
    Carga y mantiene en caché la pipeline de estimación de profundidad en memoria.
    """
    return pipeline("depth-estimation", model=model_name)


def estimate_depth(
    image: Image.Image,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    max_eval_dim: int = 640
) -> tuple[np.ndarray, Image.Image]:
    """
    Estima la profundidad relativa de una imagen dada con máxima velocidad de ejecución.
    
    Parámetros:
        image: Imagen PIL de entrada.
        model_name: Nombre del modelo en Hugging Face Hub (default: depth-anything/Depth-Anything-V2-Small-hf).
        max_eval_dim: Dimensión máxima para la evaluación del modelo IA (preserva resolución nativa al exportar).
        
    Retorna:
        depth_norm: Matriz NumPy 2D float32 normalizada en rango [0.0, 1.0].
        colormap_img: Imagen PIL con la visualización en mapa de calor (Inferno).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    orig_w, orig_h = image.size
    
    # 1. Pre-resizing optimizado para inferencia ultra rápida en CPU/GPU
    if max(orig_w, orig_h) > max_eval_dim:
        scale = max_eval_dim / float(max(orig_w, orig_h))
        eval_w, eval_h = int(orig_w * scale), int(orig_h * scale)
        eval_img = image.resize((eval_w, eval_h), Image.Resampling.BILINEAR)
    else:
        eval_img = image
        
    pipe = get_depth_pipeline(model_name)
    
    # Inferencia sin cálculo de gradientes (Torch no_grad para máxima velocidad)
    with torch.no_grad():
        result = pipe(eval_img)
        
    depth_map_pil = result["depth"]
    
    # Redimensionar al tamaño nativo exacto de la imagen original
    if depth_map_pil.size != (orig_w, orig_h):
        depth_map_pil = depth_map_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        
    depth_array = np.array(depth_map_pil, dtype=np.float32)
    
    # Normalización min-max [0.0, 1.0] (1.0 = cercano / primer plano)
    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_array, dtype=np.float32)
        
    # Mapa de calor colormap Matplotlib Inferno
    colormap_rgba = cm.inferno(depth_norm)
    colormap_rgb = (colormap_rgba[:, :, :3] * 255).astype(np.uint8)
    colormap_img = Image.fromarray(colormap_rgb)
    
    return depth_norm, colormap_img
