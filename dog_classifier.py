from classifier import classifier_network
import sys
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

root = tk.Tk()
root.title("Dog breed classifier")

height = 800
width = 800

root.geometry(str(height) + "x" + str(width))
root.resizable(False, False)

loading_window = tk.Tk()
loading_window.title("Loading dog breed classifier...")
loading_window.resizable(False, False)
loading_message = tk.Label(loading_window, text="Please wait while model is loading.", font="Times 24 bold").pack()
loading_window.update_idletasks()
loading_window.update()
cn = classifier_network()
loading_window.destroy()

def display_image(img_path):
    image_width = 640
    image_height = 480

    image = Image.open(img_path)
    image = image.resize((image_width , image_height), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    root.image = image # prevent garbage being collected
    image_canvas = tk.Canvas(root, width=image_width , height=image_height)
    image_canvas.create_image((0, 0), image=image, anchor='nw')
    image_canvas.place(x=width/2, y=height/2, anchor='center')

def classify_and_show_image(image_path):
    display_image(image_path)
    if cn.face_detector(image_path) and not cn.dog_detector(image_path):
        top_message.config(text="Hello human!")
        prediction_message.config(text="You look like a {} dog!".format(cn.inception_predict_breed(image_path)))
    elif cn.dog_detector(image_path) and not cn.face_detector(image_path):
        top_message.config(text="Hello dog!")
        prediction_message.config(text="You are a {} dog! {}/10".format(cn.inception_predict_breed(image_path), np.random.randint(low=10, high=15, size=None)))
    else:
        top_message.config(text="Hello thing!")
        prediction_message.config(text="I don't know what you are, try another picture please! If I am wrong then you might be a {}".format(cn.inception_predict_breed(image_path)))

def get_file():
    file_name = filedialog.askopenfilename(title="Choose picture to classify", filetypes=( ("jpeg files", "*.jpg"), ("png files", "*.png")) )

    if file_name == "":
        messagebox.showerror(title="Error", message="No file choosen.")
    else:
        classify_and_show_image(file_name)

button = tk.Button(root, text="Open File", width=25, command=get_file).place(x=width/2, y=height-15, anchor='center')
welcome_message = tk.Label(root, text="Hello! Please click the button below to choose a file to classify.")
welcome_message.place(x=width/2, y=7, anchor='center')

top_message = tk.Label(root)
top_message.place(x=width/2, y=height/2-250, anchor='center')

prediction_message = tk.Label(root)
prediction_message.place(x=width/2, y=height-60, anchor='center')

root.mainloop()
