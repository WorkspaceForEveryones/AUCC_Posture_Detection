import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling App")

        self.image_folder = "./dada_dataset/dada_image"

        self.image_list = sorted([f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        self.current_index = 0

        self.create_widgets()

    def create_widgets(self):
        # Display the image
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=0, column=0, columnspan=5)

        self.load_image()

        # Buttons for labeling
        for i in range(5):
            button = tk.Button(self.root, text=str(i), command=lambda i=i: self.label_image(i))
            button.grid(row=1, column=i, padx=5, pady=5)

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        img = Image.open(image_path)
        img = img.resize((300, 450))  # Use ANTIALIAS here
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def label_image(self, label):
        with open("labels.txt", "a") as f:
            f.write(f"{label}\n")

        # Move to the next image
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.root.quit()  # Close the application if all images are labeled
        else:
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
