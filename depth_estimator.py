"""
Módulo de Estimación de Profundidad (Depth Estimation)
Utiliza modelos de visión por computadora (DPT/MiDaS de Hugging Face) para predecir la profundidad relativa
de cada píxel en la imagen. Usa Matplotlib colormaps para visualizaciones sin dependencias DLL.
"""

import numpy as np
import matplotlib.cm as cm
from PIL import Image
from transformers import pipeline

_depth_pipeline = None

def get_depth_pipeline():
    """
    Carga y mantiene en caché la pipeline de estimación de profundidad.
    """
    global _depth_pipeline
    if _depth_pipeline is None:
        # Usa intel/dpt-hybrid-midas por su equilibrio entre velocidad en CPU y precisión
        _depth_pipeline = pipeline("depth-estimation", model="intel/dpt-hybrid-midas")
    return _depth_pipeline

def estimate_depth(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    Estima la profundidad relativa de una imagen dada.
    
    Parámetros:
        image: Imagen PIL de entrada.
        
    Retorna:
        depth_norm: Matriz NumPy 2D float32 normalizada en rango [0.0, 1.0].
                    1.0 representa el plano más cercano (primer plano),
                    0.0 representa el plano más lejano (segundo plano/fondo).
        colormap_img: Imagen PIL con la visualización en mapa de calor (Inferno).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    pipe = get_depth_pipeline()
    result = pipe(image)
    depth_map_pil = result["depth"]
    
    # Redimensionar al tamaño original de la imagen si difiere
    if depth_map_pil.size != image.size:
        depth_map_pil = depth_map_pil.resize(image.size, Image.Resampling.BILINEAR)
        
    depth_array = np.array(depth_map_pil, dtype=np.float32)
    
    # Normalizar entre 0.0 y 1.0 (1.0 = cercano / primer plano)
    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_array, dtype=np.float32)
        
    # Crear colormap visual con Matplotlib Inferno (Amarillo/Rojo = Cercano, Morado/Oscuro = Lejano)
    colormap_rgba = cm.inferno(depth_norm) # Retorna rango [0.0, 1.0] (H, W, 4)
    colormap_rgb = (colormap_rgba[:, :, :3] * 255).astype(np.uint8)
    colormap_img = Image.fromarray(colormap_rgb)
    
    return depth_norm, colormap_img
