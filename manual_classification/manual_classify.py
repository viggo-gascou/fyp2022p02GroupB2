import tkinter as tk
import pandas as pd
from tkinter import ttk
from os import makedirs
from PIL import Image, ImageTk


def benign():
    # Using global variable i to modify that value outside the function
    global i
    # Modifying global variable img so the image isn't garbage collected
    global img
    # Modify benign value for image in the data frame
    df.loc[df["img"] == img_names[i], "benign"] = True
    i += 1
    # Chanching the image of the canvas' image container
    img = ImageTk.PhotoImage(Image.open(img_path + img_names[i]))
    canvas.itemconfig(img_container, image=img)


def melanoma():
    # Using global variable i to modify that value outside the function
    global i
    # Modifying global variable img so the image isn't garbage collected
    global img
    # Modify benign value for image in the data frame
    df.loc[df["img"] == img_names[i], "melanoma"] = True
    i += 1
    # Chanching the image of the canvas' image container
    img = ImageTk.PhotoImage(Image.open(img_path + img_names[i]))
    canvas.itemconfig(img_container, image=img)


def keratosis():
    # Using global variable i to modify that value outside the function
    global i
    # Modifying global variable img so the image isn't garbage collected
    global img
    # Modify benign value for image in the data frame
    df.loc[df["img"] == img_names[i], "keratosis"] = True
    i += 1
    # Chanching the image of the canvas' image container
    img = ImageTk.PhotoImage(Image.open(img_path + img_names[i]))
    canvas.itemconfig(img_container, image=img)


def save_exit():
    df.to_csv("classifications/" + name + "_classification.csv")
    root.quit()


makedirs("classifications", exist_ok=True)

names = ["gustav", "magnus", "marie", "viggo", "frida"]
print("Input the number corresponding to the name of the person classifying images")
print("1: Gustav\n2: Magnus\n3: Marie\n4: Viggo\n5: Frida")
num = int(input("choice: "))
name = names[num - 1]

# Image directory path and load image name list
img_path = "../resized_data/example_image/"
with open("buckets/" + name + "_bucket.txt") as f:
    img_names = f.read().split()
# Create data frame for classification results
df = pd.DataFrame(img_names, columns=["img"])
df["benign"] = False
df["melanoma"] = False
df["keratosis"] = False
# Current index in image list
i = 0
# Create window frame and set geometry
root = tk.Tk()
root.title("Manual Classification")
root.resizable(False, False)
root.geometry("600x520")

# Configure dark theme
style = ttk.Style(root)
root.tk.call("source", "darktheme/azuredark.tcl")
style.theme_use("azure")
style.configure("Accentbutton", foreground="white")
style.configure("Togglebutton", foreground="white")

# Create image canvas and add image
canvas = tk.Canvas(root, width=600, height=450)
canvas.pack()
img = ImageTk.PhotoImage(Image.open(img_path + img_names[i]))
# Create buttons
benign_button = ttk.Button(
    root, style="Accentbutton", text="Benign", command=lambda: benign()
)
benign_button.pack(side=tk.LEFT, ipadx=5, ipady=5, expand=True)
melanoma_button = ttk.Button(
    root, style="Accentbutton", text="Melanoma", command=lambda: melanoma()
)
melanoma_button.pack(side=tk.LEFT, ipadx=5, ipady=5, expand=True)
keratosis_button = ttk.Button(
    root, style="Accentbutton", text="Keratosis", command=lambda: keratosis()
)
keratosis_button.pack(side=tk.LEFT, ipadx=5, ipady=5, expand=True)
exit_button = ttk.Button(
    root, style="Accentbutton", text="Exit and save data", command=lambda: save_exit()
)
exit_button.pack(side=tk.LEFT, ipadx=5, ipady=5, expand=True)
# Create image container on canvas
img_container = canvas.create_image(0, 0, anchor="nw", image=img)
# Run tk loop
root.mainloop()
