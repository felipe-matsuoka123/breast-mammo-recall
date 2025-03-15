import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt

def apply_window(image, window_center, window_width):
    min_value = window_center - (window_width / 2)
    max_value = window_center + (window_width / 2)
    windowed_image = np.clip(image, min_value, max_value)
    windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255.0
    return windowed_image.astype(np.uint8)

def invert_if_needed(pixels, dicom):
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        return np.max(pixels) - pixels
    return pixels

def create_breast_mask(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return mask

def crop_breast(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def apply_soft_tissue_window(image):
    # Defina limites usando percentis para capturar melhor a faixa de interesse
    lower = np.percentile(image, 10)  # ajuste conforme necessário
    upper = np.percentile(image, 90)
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    windowed = windowed.astype(np.uint8)
    
    # Aplique CLAHE para melhorar o contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    windowed = clahe.apply(windowed)
    return windowed

def normalize_image(image):
    image = image.astype(np.float32)
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

def create_3_channel_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path, force=True)
    raw_pixels = dicom.pixel_array.astype(np.float32)
    rescale_slope = getattr(dicom, "RescaleSlope", 1)
    rescale_intercept = getattr(dicom, "RescaleIntercept", 0)
    scaled_pixels = raw_pixels * rescale_slope + rescale_intercept

    scaled_pixels = invert_if_needed(scaled_pixels, dicom)

    mask_windowed_image = apply_window(scaled_pixels, window_center=128, window_width=256)
    breast_mask = create_breast_mask(mask_windowed_image)
    full_range_window = apply_window(scaled_pixels, window_center=np.min(scaled_pixels), window_width=np.max(scaled_pixels) - np.min(scaled_pixels))
    microcalcification_window = apply_window(scaled_pixels, window_center=np.percentile(scaled_pixels, 98), window_width=(np.max(scaled_pixels) - np.percentile(scaled_pixels, 98)) * 0.5)
    soft_tissue_window = apply_soft_tissue_window(scaled_pixels)
    full_range_cropped = crop_breast(full_range_window, breast_mask)
    microcalcification_cropped = crop_breast(microcalcification_window, breast_mask)
    soft_tissue_cropped = crop_breast(soft_tissue_window, breast_mask)
    full_range_cropped = normalize_image(full_range_cropped)
    microcalcification_cropped = normalize_image(microcalcification_cropped)
    soft_tissue_cropped = normalize_image(soft_tissue_cropped)
    stacked_image = np.stack([full_range_cropped, microcalcification_cropped, soft_tissue_cropped], axis=-1)
    stacked_image = cv2.resize(stacked_image, (512, 512), interpolation=cv2.INTER_AREA)
    return stacked_image


def process_dicom_dataset(input_dir, output_dir):
    dicom_files = glob(os.path.join(input_dir, "**", "*.dicom"), recursive=True)
    for dicom_path in dicom_files:
        rel_path = os.path.relpath(dicom_path, input_dir)
        output_path = os.path.join(output_dir, Path(rel_path).with_suffix(".png"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stacked_image = create_3_channel_image(dicom_path)
        cv2.imwrite(output_path, stacked_image)
        print(f"Saved: {output_path}")

def plot_result_image(image, title="Processed DICOM Image"):
    if image.ndim == 2:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        num_channels = image.shape[-1]
        fig, axs = plt.subplots(1, num_channels, figsize=(8 * num_channels, 8))
        if num_channels == 1:
            axs = [axs]
        
        for i in range(num_channels):
            axs[i].imshow(image[..., i], cmap='gray')
            axs[i].set_title(f"{title} - Canal {i+1}")
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    input_directory = "/media/felipe-matsuoka/FelipeSSD/datasets/physionet.org/files/vindr-mammo/1.0.0/images/"
    output_directory = "/media/felipe-matsuoka/FelipeSSD/datasets/vindr-mammo-png/"

    #weird_dicom_path = "/media/felipe-matsuoka/FelipeSSD/datasets/physionet.org/files/vindr-mammo/1.0.0/images/0a79f14f8d160fc88e095f92d964a03b/369cd7efe542218cec9226b8b8d4f5ad.dicom"
    #weird_image = create_3_channel_image(weird_dicom_path)
    #plot_result_image(weird_image)
    process_dicom_dataset(input_directory, output_directory)