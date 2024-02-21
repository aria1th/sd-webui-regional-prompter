# Handler for intermediate image color conversion.
# We define RGB to custom color conversion.
# Input image may contain the deterministic colors or arbitrary colors.
from typing import List, Tuple
from PIL import Image
import colorsys
import numpy as np
import cv2

def deterministic_colours(n, lcol = None):
    """Generate n visually distinct & consistent colours as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    Goes around the cyclical 0-256 and picks each /2 value for every round.
    Continuation rules: If pcyv != ccyv in next round, then we don't care.
    If pcyv == ccyv, we want to get the cval + delta of last elem.
    If lcol > n, will return it as is.
    """
    if n <= 0:
        return None
    pcyc = -1
    cval = 0
    if lcol is None:
        st = 0
    elif n <= len(lcol):
        # return lcol[:n] # Truncating the list is accurate, but pointless.
        return lcol
    else:
        st = len(lcol)
        if st > 0:
            pcyc = np.ceil(np.log2(st))
            # This is erroneous on st=2^n, but we don't care.
            dlt = 1 / (2 ** pcyc)
            cval = dlt + 2 * dlt * (st % (2 ** (pcyc - 1)) - 1)

    lhsv = []
    for i in range(st,n):
        ccyc = np.ceil(np.log2(i + 1))
        if ccyc == 0: # First col = 0.
            cval = 0
            pcyc = ccyc
        elif pcyc != ccyc: # New cycle, start from the half point between 0 and first point.
            dlt = 1 / (2 ** ccyc)
            cval = dlt
            pcyc = ccyc
        else:
            cval = cval + 2 * dlt # Jumps over existing vals.
        lhsv.append(cval)
    lhsv = [(v, 0.5, 0.5) for v in lhsv] # Hsv conversion only works 0:1.
    lrgb = [colorsys.hsv_to_rgb(*hsv) for hsv in lhsv]
    lrgb = (np.array(lrgb) * (255 + 1)).astype(np.uint8) # Convert to colour uints.
    lrgb = lrgb.reshape(-1, 3)
    if lcol is not None:
        lrgb = np.concatenate([lcol, lrgb])
    return lrgb
def generate_unique_colours(n):
    """Generate n visually distinct colors as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    """
    hsv_colors = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb_colors = [[int(i * 255) for i in colorsys.hsv_to_rgb(*hsv)] for hsv in hsv_colors]
    return np.array(rgb_colors, dtype=np.uint8)
def detect_colors(image:np.array, max_color_count = 3):
    """
    Detects the colors in the image.
    Args:
        image: np.array, input image.
        max_color_count: int, maximum number of colors to be detected.
    Returns:
        list of tuples: list of tuples of colors and their counts.
    """
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten the image
    pixels = image.reshape(-1, 3)
    # Convert to float
    pixels = np.float32(pixels)
    # Define the criteria, number of clusters(K) and apply ksmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    k = max_color_count
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    # Flatten the labels array
    labels = labels.flatten()
    # Count the frequency of each label
    counts = np.bincount(labels)
    # Sort the labels by the frequency
    sorted_labels = np.argsort(counts)
    # Sort the centers according to the labels
    centers = centers[sorted_labels]
    # Convert the centers to a list of tuples
    colors = [tuple(center[::-1]) for center in centers] # BGR to RGB
    # Return the colors and their counts
    return list(zip(colors, counts[sorted_labels]))[::-1]

def cleanup_image(image:np.array, colors:List[Tuple[int]], threshold : int=60) -> np.array:
    """
    Except the allowed colors, all other colors are converted to white (255, 255, 255).
    Args:
        array: np.array, input image.
        colors: list of tuples, allowed colors.
    Returns:
        np.array: output image.
    """
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten the image
    pixels = image.reshape(-1, 3)
    # Convert to float
    pixels = np.float32(pixels)
    # Convert the colors to a numpy array
    colors = np.array([list(color) for color in colors])
    # Calculate the distances of each pixel to the colors
    distances = np.sqrt(np.sum((pixels[:, np.newaxis] - colors) ** 2, axis=2))
    # Find the closest color
    closest_color = np.argmin(distances, axis=1)
    # Calculate the distance to the closest color
    distance = distances[np.arange(len(closest_color)), closest_color]
    # Convert the pixels to white if the distance is greater than the threshold, else to the original color
    pixels[distance > threshold] = [255, 255, 255]
    pixels[distance <= threshold] = colors[closest_color[distance <= threshold]]
    # Convert the pixels to uint8
    pixels = np.uint8(pixels)
    # Reshape the pixels to the original image shape
    image = pixels.reshape(image.shape)
    # Convert the image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def count_array_colors(array:np.array) -> dict:
    """
    Counts the colors in the image.
    """
    # Flatten the image
    pixels = array.reshape(-1, 3)
    # Convert to float
    pixels = np.float32(pixels)
    # Convert the pixels to uint8
    pixels = np.uint8(pixels)
    # Convert the pixels to a list of tuples
    pixels = [tuple(pixel) for pixel in pixels]
    # Count the frequency of each pixel
    counts = {}
    for pixel in pixels:
        if pixel in counts:
            counts[pixel] += 1
        else:
            counts[pixel] = 1
    return counts

def convert_colors(image, color_count=3, threshold=60):
    """
    Converts the colors in the image to the allowed colors.
    Args:
        image: PIL.Image.Image, input image.
        color_count: int, maximum number of colors to be detected.
        threshold: int, threshold for color conversion.
    Returns:
        PIL.Image.Image: output image.
    """
    # Convert the image to a numpy array
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Detect the colors in the image
    colors = detect_colors(image, color_count)
    # Cleanup the image
    image = cleanup_image(image, [color[0] for color in colors], threshold)
    # Convert the image to a PIL image
    color_to_use = deterministic_colours(color_count - 1) # except white
    from_colors = sorted(color[0] for color in colors)
    mapping = {from_color: to_color for from_color, to_color in zip(from_colors, color_to_use)}
    # apply the mapping
    for from_color, to_color in mapping.items():
        mask = np.all(image == from_color[::-1], axis=-1)
        print(f"Converting {from_color} to {to_color}")
        image[mask] = to_color
    # finally, convert white to black
    mask = np.all(image == [255, 255, 255], axis=-1)
    image = np.where(mask[..., None], [0, 0, 0], image)
    return image
