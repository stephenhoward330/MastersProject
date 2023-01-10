import sys

import cv2
import numpy as np

from PyQt5.QtCore import QSize, Qt, QPointF
from PyQt5.QtGui import QPainter, QTransform, QColor, QImage, QPixmap
from PyQt5.QtWidgets import *
import random

# SCALE = 1.0
#
#
# class Box(QWidget):
#     def __init__(self, points=None):
#         super(QWidget, self).__init__()
#         self.setMinimumSize(600, 600)
#         self.data_range = {'x': [-SCALE, SCALE], 'y': [-SCALE, SCALE]}
#         if points is None:
#             self.points = self.generate_random_points()
#         else:
#             self.points = points
#         self.update()
#
#     def generate_random_points(self, num_points=100):
#         pt_list = []
#         xr = self.data_range['x']
#         yr = self.data_range['y']
#         for _ in range(num_points):
#             x = random.uniform(0.0, 1.0)
#             y = random.uniform(0.0, 1.0)
#             # if not x in known_xvals:
#             #     known_xvals[x] = True
#             x_val = xr[0] + (xr[1] - xr[0]) * x
#             y_val = yr[0] + (yr[1] - yr[0]) * y
#             pt_list.append(QPointF(x_val, y_val))
#         return pt_list
#
#     def get_scale(self):
#         xr = self.data_range['x']
#         yr = self.data_range['y']
#         w = self.width()
#         h = self.height()
#         w2h_desired_ratio = (xr[1] - xr[0]) / (yr[1] - yr[0])
#         if w / h < w2h_desired_ratio:
#             scale = w / (xr[1] - xr[0])
#         else:
#             scale = h / (yr[1] - yr[0])
#         return scale
#
#     def paintEvent(self, event):
#         scale = self.get_scale()
#
#         painter = QPainter(self)
#         transform = QTransform()
#         transform.translate(self.width() / 2.0, self.height() / 2.0)
#         transform.scale(1.0, -1.0)
#         painter.setTransform(transform)
#         painter.setPen(QColor(0, 0, 0))
#         for point in self.points:
#             pt = QPointF(scale * point.x(), scale * point.y())
#             painter.drawEllipse(pt, 1.0, 1.0)

WIDTH = 600
HEIGHT = 500


def get_blank(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def generate_random_points(num_points):
    pt_list = []
    for _ in range(num_points):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        pt_list.append((x, y))
    return pt_list


def draw_points(canvas, points):
    for pt in points:
        canvas = cv2.circle(canvas, (pt[0], pt[1]), 2, (0, 0, 0), -1)
    return canvas


def get_random_image(num_points, flip=False):
    canvas = get_blank(HEIGHT, WIDTH)
    points = generate_random_points(num_points)
    # points = [(10, 10), (400, 400), (10, 400)]
    canvas = draw_points(canvas, points)
    if flip:
        canvas = cv2.flip(canvas, 0)
    return canvas


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Voronoi Art Maker")
        layout = QVBoxLayout()
        main_widget = QWidget()

        # Set the central widget of the Window
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # add the image frame
        self.image_frame = QLabel()
        layout.addWidget(self.image_frame)

        # add the 'number of points' text box and button
        h = QHBoxLayout()
        h.addWidget(QLabel('Number of Points: '))
        self.num_points = QLineEdit('100')
        self.num_points.setFixedWidth(100)
        self.num_points.textChanged.connect(self.check_input)
        h.addWidget(self.num_points)
        self.points_button = QPushButton("Generate New Points")
        self.points_button.clicked.connect(self.generate_points_clicked)
        h.addWidget(self.points_button)
        layout.addLayout(h)

        # add the 'generate voronoi' button
        self.voronoi_button = QPushButton("Generate Voronoi Diagram")
        self.voronoi_button.clicked.connect(self.voronoi_clicked)
        layout.addWidget(self.voronoi_button)

        # add a random image to the frame
        image = get_random_image(100, flip=True)
        image = QImage(image.data, WIDTH, HEIGHT, QImage.Format_RGB888)  # .rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(image))

        # show the window
        self.show()

    def generate_points_clicked(self):
        print("generate points clicked")
        image = get_random_image(int(self.num_points.text()), flip=True)
        image = QImage(image.data, WIDTH, HEIGHT, QImage.Format_RGB888)  # .rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(image))

    def voronoi_clicked(self):
        print("voronoi clicked")

    def check_input(self):
        if self.num_points.text().isdigit():
            self.points_button.setEnabled(True)
        else:
            self.points_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
