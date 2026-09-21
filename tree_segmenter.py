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

def get_salient_mask(image: Image.Image) -> np.ndarray:
    """
    Obtiene la máscara saliente de la imagen utilizando U2-Net / rembg.
    U2-Net aísla automáticamente objetos verticales (árboles) y descarta superficies planas como el suelo.
    
    Retorna:
        salient_mask: Matriz 2D uint8 (0 a 255).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    out_rgba = remove(image)
    alpha_channel = np.array(out_rgba)[:, :, 3]
    return alpha_channel

def remove_ground_plane(
    combined_bool: np.ndarray,
    depth_norm: np.ndarray,
    salient_alpha: np.ndarray,
    ground_sensitivity: float = 0.5
) -> np.ndarray:
    """
    Filtra y elimina el suelo/terreno de la máscara de primer plano.
    
    El suelo suele caracterizarse por estar en la parte inferior de la imagen,
    tener una pendiente suave de profundidad (gradiente vertical) y una baja puntuación
    de saliencia estructural en comparación con el árbol.
    """
    if ground_sensitivity <= 0.0 or not np.any(combined_bool):
        return combined_bool
        
    height, width = combined_bool.shape
    clean_bool = combined_bool.copy()
    
    # 1. Filtro por Saliencia del Suelo:
    # El suelo cercano suele dar alta profundidad pero baja saliencia de objeto.
    # Excluir píxeles donde la saliencia sea muy baja pero la profundidad alta si están en el tercio inferior
    salient_norm = salient_alpha / 255.0
    y_indices, _ = np.indices((height, width))
    y_norm = y_indices / float(height) # 0.0 arriba, 1.0 en la base
    
    # Píxeles en la mitad/base inferior de la imagen donde la saliencia es baja (< umbral)
    saliency_cutoff = 0.15 + (ground_sensitivity * 0.25)
    lower_region = (y_norm > 0.4)
    
    # Identificar candidato a suelo por falta de saliencia de objeto en la zona inferior
    ground_by_saliency = lower_region & (salient_norm < saliency_cutoff) & (depth_norm >= 0.2)
    clean_bool[ground_by_saliency] = False
    
    # 2. Filtro de Conexión a la Borde Inferior (Suelo Continuo)
    # Detectar componentes en la máscara que toquen el borde inferior pero no tengan suficiente altura vertical
    labeled_mask = label(clean_bool)
    regions = regionprops(labeled_mask)
    
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        
        # Si el componente toca el borde inferior (max_row cerca del final)
        touches_bottom = (max_row >= height - int(height * 0.03))
        aspect_ratio = region_width / max(1, region_height)
        
        # Si el componente toca la base y es plano/ancho (aspect ratio alto) o de poca altura relativa
        is_flat_ground = touches_bottom and (aspect_ratio > (2.5 - ground_sensitivity) or region_height < height * 0.25)
        
        # O si el centroide del componente está muy abajo y tiene baja saliencia promedio
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
    ground_sensitivity: float = 0.5
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
        
    Retorna:
        binary_mask: Matriz 2D uint8 (255 = Árbol primer plano, 0 = Fondo/Segundo plano/Suelo).
        stats: Diccionario con estadísticas de la segmentación.
    """
    height, width = depth_norm.shape
    
    # 1. Máscara por Umbral de Profundidad (Primer Plano)
    depth_mask = (depth_norm >= depth_threshold)
    
    # 2. Máscara por Objetos Salientes (U2-Net)
    salient_alpha = get_salient_mask(image)
    if salient_alpha.shape != (height, width):
        salient_img = Image.fromarray(salient_alpha).resize((width, height), Image.Resampling.BILINEAR)
        salient_alpha = np.array(salient_img)
        
    salient_bool = (salient_alpha > 30)
    
    # 3. Fusión Adaptativa
    if saliency_weight > 0.0:
        depth_weight = 1.0 - saliency_weight
        fused_score = (depth_norm * depth_weight) + ((salient_alpha / 255.0) * saliency_weight)
        combined_bool = (fused_score >= depth_threshold)
    else:
        combined_bool = depth_mask.copy()
        
    # 4. EXCLUSIÓN DE SUELO / TERRENO (NUEVA FUNCIONALIDAD)
    if filter_ground:
        combined_bool = remove_ground_plane(
            combined_bool=combined_bool,
            depth_norm=depth_norm,
            salient_alpha=salient_alpha,
            ground_sensitivity=ground_sensitivity
        )
        
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
