import tkinter as tk
from tkinter import filedialog
from models import mtcnn_wrapper, viola_jones
from utils import comparator

def upload_image(use_mtcnn: bool):
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not path:
        return
    if use_mtcnn:
        mtcnn_wrapper.mtcnn_face_detect(path)
    else:
        viola_jones.viola_jones_face_detect(path)

def open_webcam(use_mtcnn: bool):
    if use_mtcnn:
        mtcnn_wrapper.mtcnn_face_detect_webcam()
    else:
        viola_jones.viola_jones_face_detect_webcam()

def compare():
    comparator.compare()


def main():
    root = tk.Tk()
    root.title("Face Detection")
    tk.Button(root, text="Upload Image MTCNN", font=("Arial", 14), command=lambda: upload_image(use_mtcnn=True)).pack(pady=10)
    tk.Button(root, text="Open Webcam MTCNN", font=("Arial", 14), command=lambda: open_webcam(use_mtcnn=True)).pack(pady=10)
    tk.Button(root, text="Upload Image Viola-Jones", font=("Arial", 14), command=lambda: upload_image(use_mtcnn=False)).pack(pady=10)
    tk.Button(root, text="Open Webcam Viola-Jones", font=("Arial", 14), command=lambda: open_webcam(use_mtcnn=False)).pack(pady=10)
    tk.Button(root, text="Compare", font=("Arial", 14), command=lambda: compare()).pack(pady=10)

    canvas = tk.Canvas(root)
    canvas.pack()

    root.mainloop()