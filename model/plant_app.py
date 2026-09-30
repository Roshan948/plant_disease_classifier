import json
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

img_size = (128, 128)
model = tf.keras.models.load_model('plant_disease_model.keras')
class_names = json.load(open('class_names.json'))


def clean(name):
    return name.replace('Tomato___', '').replace('_', ' ')


def predict(path):
    img = np.array(Image.open(path).convert('RGB'))
    arr = tf.image.resize(img, img_size).numpy()[None]
    probs = model.predict(arr, verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    return [(clean(class_names[i]), float(probs[i]) * 100) for i in top]


root = tk.Tk()
root.title('Tomato Leaf Disease Detector')
root.geometry('340x470')

panel = tk.Label(root, text='No image selected', width=36, height=14, relief='groove')
panel.pack(pady=15)

result = tk.Label(root, text='', font=('Arial', 12), justify='left')
result.pack(pady=10)


def choose():
    path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG')])
    if not path:
        return
    photo = ImageTk.PhotoImage(Image.open(path).convert('RGB').resize((250, 250)))
    panel.config(image=photo, text='', width=250, height=250)
    panel.image = photo
    lines = [f'{name}: {conf:.1f}%' for name, conf in predict(path)]
    result.config(text='\n'.join(lines))


tk.Button(root, text='Choose leaf image', command=choose, width=20).pack(pady=5)

root.mainloop()
