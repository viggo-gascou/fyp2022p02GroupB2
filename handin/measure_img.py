from measure import measure
import matplotlib.pyplot as plt
import sys
import numpy as np

img_file = sys.argv[1]
img = plt.imread(f"data_to_resize/example_image/{img_file}.jpg")
seg = plt.imread(f"data_to_resize/example_segmentation/{img_file}_segmentation.png")
print(measure(img, seg))
