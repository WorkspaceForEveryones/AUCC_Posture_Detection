import matplotlib.pyplot as plt
import os
import keyboard
import tkinter as tk


class PointRecorder:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = sorted([os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith((".jpg", ".png"))])
        self.points_data = {}
        self.current_image = None
        self.counter = 0

    def load_image(self, image_path):
        self.current_image = image_path
        if self.current_image not in self.points_data:
            self.points_data[self.current_image] = []
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(plt.imread(image_path))
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes == self.ax:
            self.counter += 1
            x, y = event.xdata, event.ydata
            self.points_data[self.current_image].append((x, y))
            self.ax.plot(x, y, 'ro')
            self.ax.annotate(f'Point {self.counter}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for image, points in self.points_data.items():
                f.write(', '.join([f'{point[0]}, {point[1]}' for point in points]))
                f.write('\n')

    def show_plot(self):
        plt.show()

if __name__ == "__main__":
    image_folder = 'new_dataset/new_image'
    output_file = 'new_dataset/correct_coord_file.txt'

    recorder = PointRecorder(image_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            recorder.load_image(image_path)
            recorder.show_plot()

    recorder.save_to_file(output_file)
    print(f"Points recorded to {output_file}")
