"""
Módulo de Utilidades (Utils)
Proporciona funciones para generar superposiciones de color, recortes transparentes,
exportación de imágenes y generación de imágenes de prueba sintéticas.
"""

import io
import numpy as np
from PIL import Image, ImageDraw

def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.45
) -> Image.Image:
    """
    Crea una superposición de color semi-transparente sobre la imagen original usando la máscara.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_np = np.array(image, dtype=np.uint8)
    h, w, _ = img_np.shape
    
    if mask.shape != (h, w):
        mask_pil = Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST)
        mask = np.array(mask_pil)
        
    overlay_np = img_np.astype(np.float32).copy()
    fg_indices = (mask > 128)
    
    color_arr = np.array(color_rgb, dtype=np.float32)
    overlay_np[fg_indices] = (img_np[fg_indices] * (1.0 - alpha)) + (color_arr * alpha)
    
    return Image.fromarray(overlay_np.astype(np.uint8))

def extract_cutout(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Extrae el objeto de primer plano aislando el árbol con fondo transparente (RGBA).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_np = np.array(image)
    h, w, _ = img_np.shape
    
    if mask.shape != (h, w):
        mask_pil = Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST)
        mask = np.array(mask_pil)
        
    alpha_channel = np.where(mask > 128, 255, 0).astype(np.uint8)
    rgba_np = np.dstack([img_np, alpha_channel])
    return Image.fromarray(rgba_np, mode="RGBA")

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convierte una imagen PIL a bytes para permitir su descarga en Streamlit.
    """
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def create_synthetic_sample_trees() -> dict[str, Image.Image]:
    """
    Genera dos imágenes sintéticas de demostración con árboles en primer plano y segundo plano.
    """
    width, height = 800, 600
    
    # Imagen 1: Árbol en Primer Plano Grande + Árboles en Segundo Plano Lejanos
    img1 = Image.new("RGB", (width, height), (135, 206, 235))
    draw1 = ImageDraw.Draw(img1)
    
    # Fondo
    draw1.ellipse([-100, 300, 900, 700], fill=(70, 130, 70))
    
    # Árboles lejanos (segundo plano)
    draw1.rectangle([150, 260, 180, 380], fill=(80, 50, 20))
    draw1.ellipse([110, 180, 220, 300], fill=(40, 90, 40))
    
    draw1.rectangle([600, 250, 630, 380], fill=(80, 50, 20))
    draw1.ellipse([550, 170, 680, 290], fill=(35, 85, 35))
    
    # Árbol Principal (Primer Plano)
    draw1.rectangle([340, 240, 420, 580], fill=(101, 67, 33))
    draw1.ellipse([230, 80, 530, 380], fill=(34, 139, 34))
    draw1.ellipse([270, 50, 490, 280], fill=(46, 170, 46))
    draw1.ellipse([190, 140, 400, 330], fill=(28, 120, 28))
    draw1.ellipse([360, 130, 560, 340], fill=(50, 185, 50))
    
    # Imagen 2: Bosque con Árbol Frondoso en Primer Plano Izquierda
    img2 = Image.new("RGB", (width, height), (176, 224, 230))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([0, 350, width, height], fill=(60, 110, 50))
    
    for x_pos, h_top, scale in [(80, 200, 0.5), (220, 220, 0.6), (580, 190, 0.55), (700, 230, 0.65)]:
        trunk_w = int(20 * scale)
        draw2.rectangle([x_pos, 300, x_pos + trunk_w, 420], fill=(90, 60, 30))
        draw2.ellipse([x_pos - int(50*scale), h_top, x_pos + int(70*scale), 340], fill=(40, 100 - int(scale*20), 40))

    draw2.rectangle([120, 180, 210, 600], fill=(115, 75, 40))
    draw2.ellipse([20, 20, 310, 320], fill=(30, 145, 40))
    draw2.ellipse([60, 0, 280, 220], fill=(45, 175, 55))
    
    return {
        "Muestra 1: Árbol Principal Central": img1,
        "Muestra 2: Árbol Frondoso Izquierda": img2
    }
