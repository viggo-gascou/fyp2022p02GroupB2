from os import listdir, mkdir
from shutil import rmtree


img_names = listdir("../resized_data/example_image_resized")
img_names.sort()
indices = [i for i in range(0, 451, 90)]
indices = list(map(lambda x: x % 150, indices))

buckets = []
for i1, i2 in zip(indices, indices[1:]):
    if i2 < i1:
        bucket = img_names[i1:] + img_names[:i2]
    else:
        bucket = img_names[i1:i2]
    buckets.append(bucket)

try:
    rmtree("buckets")
except FileNotFoundError:
    pass
mkdir("buckets")
names = ["gustav", "magnus", "marie", "viggo", "frida"]
for name, bucket in zip(names, buckets):
    with open(name + "_bucket.txt", "w") as f:
        f.write(" ".join(bucket))
