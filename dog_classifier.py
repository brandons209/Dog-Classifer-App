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
loading_window.title("loading dog breed classifier...")
loading_window.resizable(False, False)
loading_message = tk.Label(loading_window, text="pwease wait while we load som files!", font="Times 24 bold").pack()
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
        top_message.config(text="henlo hooman!")
        prediction_message.config(text="u look lik a {} doggo!!".format(cn.inception_predict_breed(image_path)))
    elif cn.dog_detector(image_path) and not cn.face_detector(image_path):
        top_message.config(text="henlo doggo!")
        prediction_message.config(text="u are a {} doggo!! {}/10".format(cn.inception_predict_breed(image_path), np.random.randint(low=10, high=15, size=None)))
    else:
        top_message.config(text="henlo thingy!")
        prediction_message.config(text="i dont know what u r, plz try another pic! BUT, if i am rong (wich i somtimez am) then u might b a {}".format(cn.inception_predict_breed(image_path)))

def get_file():
    file_name = filedialog.askopenfilename(title="OwO whats this?", filetypes=( ("jpeg files", "*.jpg"), ("png files", "*.png")) )

    if file_name == "":
        messagebox.showerror(title="oopsie whoopsie!", message="uh oh! looks lik u didnt choose a file! try again!")
    else:
        classify_and_show_image(file_name)

button = tk.Button(root, text="Open File", width=25, command=get_file).place(x=width/2, y=height-15, anchor='center')
welcome_message = tk.Label(root, text="henlo! pwease gibe image for me to classify!")
welcome_message.place(x=width/2, y=7, anchor='center')

top_message = tk.Label(root)
top_message.place(x=width/2, y=height/2-250, anchor='center')

prediction_message = tk.Label(root)
prediction_message.place(x=width/2, y=height-60, anchor='center')

root.mainloop()
