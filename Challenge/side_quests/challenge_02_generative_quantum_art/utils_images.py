"""
Utility functions for loading and processing images for quantum circuits.
Supports binary (black/white), grayscale, and color image formats.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def _load_and_handle_transparency(image_path, background_color='white'):
    """
    Internal helper function to load an image and handle transparency.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    background_color : str or tuple, default='white'
        Background color for transparent areas. Can be 'white', 'black',
        or an RGB tuple like (255, 255, 255)
        Type: str | tuple[int, int, int]
    
    Returns:
    --------
    PIL.Image
        Loaded image with transparency handled (composited on specified background)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
    
    # Open image
    img = Image.open(image_path)
    
    # If image has transparency, composite it on a colored background
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Create background with specified color
        background = Image.new('RGB', img.size, background_color)  # type: ignore
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    
    return img


def _resize_image(img, size):
    """
    Internal helper function to resize an image.
    
    Parameters:
    -----------
    img : PIL.Image
        Image to resize
    size : int or tuple or None
        Target size. If int, creates square image (size x size).
        If tuple (width, height). If None, returns image unchanged.
    
    Returns:
    --------
    PIL.Image
        Resized image (or original if size is None)
    """
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.Resampling.NEAREST)
    return img


def load_binary_image(image_path, size=None, threshold=128, background_color='white'):
    """
    Load an image and convert it to a binary (black/white) matrix.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    size : int or tuple, optional
        Target size. If int, creates square image (size x size).
        If tuple (width, height). If None, keeps original size.
    threshold : int, default=128
        Threshold value (0-255). Pixels below threshold become 1 (black),
        pixels above become 0 (white).
    background_color : str or tuple, default='white'
        Background color for transparent areas. Can be 'white', 'black',
        or an RGB tuple like (255, 255, 255)
    
    Returns:
    --------
    numpy.ndarray
        Binary matrix of 0s and 1s where 1 represents black pixels
    """
    # Load image and handle transparency
    img = _load_and_handle_transparency(image_path, background_color)
    
    # Convert to grayscale
    img = img.convert("L")
    
    # Resize if size specified
    img = _resize_image(img, size)
    
    # Convert to array and apply threshold
    pixel_array = np.array(img)
    binary_matrix = (pixel_array < threshold).astype(int)
    
    return binary_matrix


def load_grayscale_image(image_path, size=None, normalize=False, background_color='white'):
    """
    Load an image and convert it to grayscale.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    size : int or tuple, optional
        Target size. If int, creates square image (size x size).
        If tuple (width, height). If None, keeps original size.
    normalize : bool, default=False
        If True, normalizes pixel values to [0, 1] range.
        If False, keeps values in [0, 255] range.
    background_color : str or tuple, default='white'
        Background color for transparent areas. Can be 'white', 'black',
        or an RGB tuple like (255, 255, 255)
    
    Returns:
    --------
    numpy.ndarray
        Grayscale matrix with values in [0, 255] or [0, 1] if normalized
    """
    # Load image and handle transparency
    img = _load_and_handle_transparency(image_path, background_color)
    
    # Convert to grayscale
    img = img.convert("L")
    
    # Resize if size specified
    img = _resize_image(img, size)
    
    # Convert to array
    pixel_array = np.array(img)
    
    # Normalize if requested
    if normalize:
        pixel_array = pixel_array / 255.0
    
    return pixel_array


def load_color_image(image_path, size=None, normalize=False, background_color='white'):
    """
    Load a color image (RGB).
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    size : int or tuple, optional
        Target size. If int, creates square image (size x size).
        If tuple (width, height). If None, keeps original size.
    normalize : bool, default=False
        If True, normalizes pixel values to [0, 1] range.
        If False, keeps values in [0, 255] range.
    background_color : str or tuple, default='white'
        Background color for transparent areas. Can be 'white', 'black',
        or an RGB tuple like (255, 255, 255)
    
    Returns:
    --------
    numpy.ndarray
        Color matrix with shape (height, width, 3) for RGB channels
        Values in [0, 255] or [0, 1] if normalized
    """
    # Load image and handle transparency
    img = _load_and_handle_transparency(image_path, background_color)
    
    # Convert to RGB
    img = img.convert("RGB")
    
    # Resize if size specified
    img = _resize_image(img, size)
    
    # Convert to array
    pixel_array = np.array(img)
    
    # Normalize if requested
    if normalize:
        pixel_array = pixel_array / 255.0
    
    return pixel_array


def display_image(pixel_array, title="Image", cmap=None, figsize=(8, 8)):
    """
    Display an image from a pixel array.
    
    Parameters:
    -----------
    pixel_array : numpy.ndarray
        Image array to display (binary, grayscale, or color)
    title : str, default="Image"
        Title for the plot
    cmap : str, optional
        Colormap to use. If None, uses 'gray' for 2D arrays and default for 3D.
        Options: 'gray', 'viridis', 'plasma', 'binary', etc.
    figsize : tuple, default=(8, 8)
        Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    
    # Auto-detect colormap if not specified
    if cmap is None:
        if pixel_array.ndim == 2:
            cmap = 'gray'
    
    plt.imshow(pixel_array, cmap=cmap, interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_grayscale_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    show: bool = True,
    verbose: bool = True,
    title_original: str = "Original",
    title_reconstructed: str = "Reconstructed",
) -> tuple[float, np.ndarray]:
    """Compare a reconstructed grayscale image to the original.

    This is intended as a debug/teaching helper (prints + optional plots).

    Behavior:
    - Ensures both images are shape-compatible (reconstructed is reshaped to original.shape if needed).
    - Clips reconstructed values to [0, 255].
    - Computes absolute error and prints Mean Absolute Error (MAE).
    - If show=True, displays 3 plots side by side:
      original, reconstructed, absolute error (same style as the notebook snippet).

    Args:
        original: 2D grayscale image (ideally uint8) with values in [0, 255].
        reconstructed: 2D image, or flat array that can be reshaped to original.shape.
        show: If True, show the comparison plots.
        verbose: If True, print ranges + MAE.
        title_original: Title for the original plot.
        title_reconstructed: Title for the reconstructed plot.

    Returns:
        mae: Mean absolute error (float).
        reconstructed_img: Reconstructed image as int array, clipped to [0,255], shape == original.shape.
        error: Absolute error image as int array, same shape.
    """
    original_img = np.asarray(original)
    if original_img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale original image, got shape {original_img.shape}.")

    reconstructed_arr = np.asarray(reconstructed)

    # Reshape if reconstruction is flat or shape-mismatched but size matches.
    if reconstructed_arr.shape != original_img.shape:
        if reconstructed_arr.size == original_img.size:
            reconstructed_arr = reconstructed_arr.reshape(original_img.shape)
        else:
            raise ValueError(
                "Reconstructed image shape is incompatible with original. "
                f"original.shape={original_img.shape}, reconstructed.shape={reconstructed_arr.shape}"
            )

    reconstructed_img = np.asarray(reconstructed_arr, dtype=int)
    reconstructed_img = np.clip(reconstructed_img, 0, 255)

    original_int = np.asarray(original_img, dtype=int)
    error = np.abs(original_int - reconstructed_img)
    mae = float(np.mean(error))

    if verbose:
        print("\n Results:")
        print(f"Original range: [{original_int.min()}, {original_int.max()}]")
        print(f"Retrieved range: [{reconstructed_img.min()}, {reconstructed_img.max()}]")
        print("\n Error analysis:")
        print(f"   Mean absolute error: {mae:.2f}")

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"{title_original}")
        axes[0].axis('off')

        axes[1].imshow(reconstructed_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"{title_reconstructed}\n(MAE={mae:.1f})")
        axes[1].axis('off')

        vmax_err = int(np.max(error)) if error.size else 0
        im_err = axes[2].imshow(error, cmap='Reds', vmin=0, vmax=max(vmax_err, 1))
        axes[2].set_title("Absolute Error")
        axes[2].axis('off')

        # Legend for the error heatmap (how red maps to magnitude)
        cbar = fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Absolute error (0–255)")

        plt.tight_layout()
        plt.show()

    return mae, error


def get_image_info(pixel_array):
    """
    Get information about an image array.
    
    Parameters:
    -----------
    pixel_array : numpy.ndarray
        Image array
    
    Returns:
    --------
    dict
        Dictionary containing image information (shape, dtype, min, max, etc.)
    """
    info = {
        'shape': pixel_array.shape,
        'dtype': pixel_array.dtype,
        'min_value': pixel_array.min(),
        'max_value': pixel_array.max(),
        'mean_value': pixel_array.mean(),
        'total_pixels': pixel_array.size
    }
    
    if pixel_array.ndim == 2:
        info['type'] = 'binary or grayscale'
        info['height'], info['width'] = pixel_array.shape
    elif pixel_array.ndim == 3:
        info['type'] = 'color (RGB)'
        info['height'], info['width'], info['channels'] = pixel_array.shape
    
    return info
