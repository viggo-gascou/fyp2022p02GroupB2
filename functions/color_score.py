import numpy as np
from PIL import Image


def color_score(img_file, seg_file):
    # Open the segmentation file as RGB and as bitmap
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
    # Convert image to array with RGB values between [0, 1]
    img_arr = np.array(img) / 255
    # Remove all black pixels
    lesion = img_arr[np.where(~(
        (img_arr[:, :, 0] == 0)
        & (img_arr[:, :, 1] == 0)
        & (img_arr[:, :, 2] == 0))
    )]
    # Count non-black pixels in lesion
    pixels = len(lesion)
    color_count = 0
    # Masks for detecting the six colors, from (Majumder & Ullah, 2019)
    masks = {
        "white": (lesion[:, 0] >= 0.8) & (lesion[:, 1] >= 0.8) & (lesion[:, 2] >= 0.8),
        "red": (lesion[:, 0] >= 0.588) & (lesion[:, 1] >= 0.2) & (lesion[:, 2] >= 0.2),
        "lightbrown": (lesion[:, 0] >= 0.588)
        & (lesion[:, 1] > 0.2)
        & (lesion[:, 2] > 0)
        & (lesion[:, 0] <= 0.94)
        & (lesion[:, 1] <= 0.588)
        & (lesion[:, 2] < 0.392),
        "darkbrown": (lesion[:, 0] > 0.243)
        & (lesion[:, 1] >= 0)
        & (lesion[:, 2] > 0)
        & (lesion[:, 0] < 0.56)
        & (lesion[:, 1] < 0.392)
        & (lesion[:, 2] < 0.392),
        "blue-gray": (lesion[:, 0] >= 0)
        & (lesion[:, 1] >= 0.392)
        & (lesion[:, 2] >= 0.490)
        & (lesion[:, 0] <= 0.588)
        & (lesion[:, 1] <= 0.588)
        & (lesion[:, 2] <= 0.588),
        "black": (lesion[:, 0] <= 0.243)
        & (lesion[:, 1] <= 0.243)
        & (lesion[:, 2] <= 0.243),
    }
    # Check for each color if amount of pixels of that color is over 5%
    # Add 1 to the color score if true
    for mask in masks.values():
        if len(lesion[mask]) / pixels > 0.05:
            color_count += 1
    return color_count
