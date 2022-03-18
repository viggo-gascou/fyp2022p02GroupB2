from os import listdir, mkdir
from os.path import isdir
from shutil import rmtree
from PIL import Image
from sys import argv
from pathlib import Path
from tqdm import tqdm


if len(argv) < 2:
    print(
        "Please provide a parent directory to resize the subdirectories of as an argument:"
    )
    print("python resize.py [path]")
else:
    data_dir = argv[1] + "/"
    if not Path(data_dir).exists():
        print("Provided path does not exist")
    elif not isdir(data_dir):
        print("Provided path is not a directory")
    else:
        try:
            rmtree("resized_data")
        except FileNotFoundError:
            pass
        mkdir("resized_data")
        subdirs = listdir(data_dir)
        subdirs = [dir for dir in subdirs if isdir(data_dir + dir)]
        paths = list(map(lambda x: data_dir + x, subdirs))
        for dir, path in zip(subdirs, paths):
            img_names = [file for file in listdir(path) if file[0] != "."]
            resized_path = f"resized_data/{dir}_resized/"
            mkdir(resized_path)
            for img_name in tqdm(img_names):
                img = Image.open(path + "/" + img_name)
                img.thumbnail((600, 600))
                img.save(resized_path + img_name)
