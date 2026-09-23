"""
Módulo de Segmentación de Árboles en Primer Plano (Foreground Tree Segmenter)
Combina detección de objetos salientes (rembg/U2-Net) con máscaras de mapa de profundidad,
filtrado de suelo/terreno y operaciones morfológicas avanzadas con scikit-image.
"""

import numpy as np
from PIL import Image
from rembg import remove
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

def get_salient_mask(image: Image.Image, max_eval_dim: int = 1024) -> np.ndarray:
    """
    Obtiene la máscara saliente de la imagen utilizando U2-Net / rembg de forma ultra rápida.
    
    Retorna:
        salient_mask: Matriz 2D uint8 (0 a 255).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    orig_w, orig_h = image.size
    
    # Pre-resizing optimizado para inferencia rápida de U2-Net
    if max(orig_w, orig_h) > max_eval_dim:
        scale = max_eval_dim / float(max(orig_w, orig_h))
        eval_w, eval_h = int(orig_w * scale), int(orig_h * scale)
        eval_img = image.resize((eval_w, eval_h), Image.Resampling.BILINEAR)
    else:
        eval_img = image
        
    out_rgba = remove(eval_img)
    alpha_channel = np.array(out_rgba)[:, :, 3]
    
    # Redimensionar al tamaño nativo de la imagen original
    if alpha_channel.shape != (orig_h, orig_w):
        alpha_img = Image.fromarray(alpha_channel).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        alpha_channel = np.array(alpha_img)
        
    return alpha_channel

def remove_ground_plane(
    combined_bool: np.ndarray,
    depth_norm: np.ndarray,
    salient_alpha: np.ndarray,
    image: Image.Image = None,
    ground_sensitivity: float = 0.5
) -> np.ndarray:
    """
    Filtra y elimina el suelo/terreno de la máscara de primer plano.
    
    El suelo suele caracterizarse por estar en la parte inferior de la imagen,
    tener espectro de color de tierra/arena desértica o pasto plano y una baja puntuación
    de saliencia estructural en comparación con el árbol.
    """
    if ground_sensitivity <= 0.0 or not np.any(combined_bool):
        return combined_bool
        
    height, width = combined_bool.shape
    clean_bool = combined_bool.copy()
    
    # Asegurar dimensiones exactas de imagen y saliencia
    if image is not None and image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        
    if salient_alpha.shape != (height, width):
        salient_img = Image.fromarray(salient_alpha).resize((width, height), Image.Resampling.BILINEAR)
        salient_alpha = np.array(salient_img)
        
    salient_norm = salient_alpha / 255.0
    y_indices, _ = np.indices((height, width))
    y_norm = y_indices / float(height) # 0.0 arriba, 1.0 en la base
    
    # 1. Filtro Espectral de Arena y Suelo Desértico/Tierra
    if image is not None:
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        cmax = np.maximum(r_n, np.maximum(g_n, b_n))
        cmin = np.minimum(r_n, np.minimum(g_n, b_n))
        delta = cmax - cmin

        h_chan = np.zeros_like(cmax)
        mask_r = (cmax == r_n) & (delta != 0)
        mask_g = (cmax == g_n) & (delta != 0)
        mask_b = (cmax == b_n) & (delta != 0)
        h_chan[mask_r] = (((g_n[mask_r] - b_n[mask_r]) / delta[mask_r]) % 6) * 60
        h_chan[mask_g] = (((b_n[mask_g] - r_n[mask_g]) / delta[mask_g]) + 2) * 60
        h_chan[mask_b] = (((r_n[mask_b] - g_n[mask_b]) / delta[mask_b]) + 4) * 60

        is_sand_ground = (r > 95) & (g > 65) & (b > 35) & (r > b * 1.10) & (h_chan >= 10) & (h_chan <= 55) & (y_norm > 0.45) & (salient_norm < 0.25)
        clean_bool[is_sand_ground] = False
    
    # 2. Filtro por Saliencia del Suelo:
    saliency_cutoff = 0.12 + (ground_sensitivity * 0.18)
    lower_region = (y_norm > 0.50)
    
    ground_by_saliency = lower_region & (salient_norm < saliency_cutoff)
    clean_bool[ground_by_saliency] = False
    
    # 3. Filtro de Conexión al Borde Inferior (Suelo Continuo)
    labeled_mask = label(clean_bool)
    regions = regionprops(labeled_mask)
    
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        
        touches_bottom = (max_row >= height - int(height * 0.03))
        aspect_ratio = region_width / max(1, region_height)
        
        is_flat_ground = touches_bottom and (aspect_ratio > (2.5 - ground_sensitivity) or region_height < height * 0.25)
        
        region_mask = (labeled_mask == region.label)
        avg_saliency = np.mean(salient_norm[region_mask])
        
        if is_flat_ground or (touches_bottom and avg_saliency < (0.2 + ground_sensitivity * 0.2)):
            clean_bool[region_mask] = False

    return clean_bool

def segment_foreground_tree(
    image: Image.Image,
    depth_norm: np.ndarray,
    depth_threshold: float = 0.45,
    saliency_weight: float = 0.5,
    morph_kernel_size: int = 5,
    min_area_filter: bool = True,
    filter_ground: bool = True,
    ground_sensitivity: float = 0.5,
    salient_alpha: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """
    Genera la máscara del árbol en primer plano combinando profundidad, saliencia y filtrado de suelo.
    
    Parámetros:
        image: Imagen PIL de entrada.
        depth_norm: Mapa de profundidad normalizado [0.0, 1.0] (1.0 = primer plano).
        depth_threshold: Umbral de profundidad mínimo para considerar primer plano [0.0, 1.0].
        saliency_weight: Peso del filtro de saliencia U2-Net (0.0 a 1.0).
        morph_kernel_size: Tamaño del radio morfológico.
        min_area_filter: Elimina componentes aislados diminutos del fondo.
        filter_ground: Activa el filtrado automático de suelo/terreno.
        ground_sensitivity: Sensibilidad de exclusión del suelo (0.0 = desactivado, 1.0 = estricto).
        salient_alpha: Máscara saliente precargada/en caché (opcional para máxima velocidad).
        
    Retorna:
        binary_mask: Matriz 2D uint8 (255 = Árbol primer plano, 0 = Fondo/Segundo plano/Suelo).
        stats: Diccionario con estadísticas de la segmentación.
    """
    height, width = depth_norm.shape
    
    # Asegurar alineación de dimensiones de la imagen de entrada con el mapa de profundidad
    if image is not None and image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        
    # 1. Máscara por Umbral de Profundidad (Primer Plano)
    depth_mask = (depth_norm >= depth_threshold)
    
    # 2. Máscara por Objetos Salientes (U2-Net)
    if salient_alpha is None:
        salient_alpha = get_salient_mask(image)
        
    if salient_alpha.shape != (height, width):
        salient_img = Image.fromarray(salient_alpha).resize((width, height), Image.Resampling.BILINEAR)
        salient_alpha = np.array(salient_img)
        
    salient_bool = (salient_alpha > 30)
    
    # 3. Puerta Estricta de Profundidad Física con Calibración Dinámica
    salient_norm = salient_alpha / 255.0
    
    if image is not None:
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        r, b = img_np[:, :, 0], img_np[:, :, 2]
        y_indices, _ = np.indices((height, width))
        y_norm = y_indices / float(height)
        is_sky = (b > r * 1.02) & (b > 90) & (y_norm < 0.65)
    else:
        is_sky = np.zeros((height, width), dtype=bool)
        
    non_sky = ~is_sky
    cutoff_depth = depth_threshold * 0.45
    depth_gate = (depth_norm >= cutoff_depth) & non_sky
    
    if saliency_weight > 0.0:
        structure_candidate = (salient_norm > (0.10 - saliency_weight * 0.08))
        combined_bool = depth_gate & structure_candidate
    else:
        combined_bool = depth_gate.copy()
        
    # 4. EXCLUSIÓN DE SUELO / TERRENO
    if filter_ground:
        combined_bool = remove_ground_plane(
            combined_bool=combined_bool,
            depth_norm=depth_norm,
            salient_alpha=salient_alpha,
            image=image,
            ground_sensitivity=ground_sensitivity
        )
        combined_bool = combined_bool & depth_gate  # Re-enforce strict physical depth gate
        
    # 5. Operaciones Morfológicas con scikit-image
    if morph_kernel_size > 1:
        radius = max(1, morph_kernel_size // 2)
        selem = disk(radius)
        combined_bool = opening(combined_bool, selem)
        combined_bool = closing(combined_bool, selem)
        
    # 6. Filtrado de Componentes Conectados Pequeños
    if min_area_filter and np.any(combined_bool):
        labeled_mask = label(combined_bool)
        regions = regionprops(labeled_mask)
        total_img_area = width * height
        min_component_area = total_img_area * 0.005  # 0.5% del área total
        
        clean_bool = np.zeros_like(combined_bool, dtype=bool)
        for region in regions:
            if region.area >= min_component_area:
                clean_bool[labeled_mask == region.label] = True
        combined_bool = clean_bool

    # Convertir máscara a uint8 (255 / 0)
    final_mask = (combined_bool.astype(np.uint8)) * 255
    
    fg_pixels = np.count_nonzero(final_mask)
    fg_percentage = (fg_pixels / (height * width)) * 100.0
    
    stats_dict = {
        "fg_percentage": fg_percentage,
        "width": width,
        "height": height,
        "depth_threshold_used": depth_threshold,
        "ground_filtered": filter_ground
    }
    
    return final_mask, stats_dict
