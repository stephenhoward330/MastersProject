import sys

import cv2
import numpy as np
import random
from tqdm import tqdm

from PyQt5.QtCore import QSize, Qt, QPointF
from PyQt5.QtGui import QPainter, QTransform, QColor, QImage, QPixmap
from PyQt5.QtWidgets import *

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

WIDTH = 300
HEIGHT = 250
DEFAULT_NUM_POINTS = 15


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
        self.show_points = True
        self.points = []

        self.setWindowTitle("Voronoi Art Maker")
        layout = QVBoxLayout()
        main_widget = QWidget()

        # Set the central widget of the Window
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # add the image frame
        self.progress = 0
        # TODO: add conditional progress bar on top of the image frame
        self.image_frame = QLabel()
        layout.addWidget(self.image_frame)

        # add the 'number of points' text box and button
        h = QHBoxLayout()
        h.addWidget(QLabel('Number of Points: '))
        self.num_points = QLineEdit(str(DEFAULT_NUM_POINTS))
        self.num_points.setFixedWidth(75)
        self.num_points.textChanged.connect(self.check_input)
        h.addWidget(self.num_points)
        self.points_button = QPushButton("Generate New Points")
        self.points_button.clicked.connect(self.generate_points_clicked)
        h.addWidget(self.points_button)
        layout.addLayout(h)

        # add the toggle points checkbox
        h = QHBoxLayout()
        h.addWidget(QLabel('Show Points? '))
        self.toggle_points_box = QCheckBox()
        self.toggle_points_box.setChecked(True)
        self.toggle_points_box.stateChanged.connect(self.toggle_points_clicked)
        h.addWidget(self.toggle_points_box)
        # add the 'generate voronoi' button
        self.voronoi_button = QPushButton("Generate Voronoi Diagram")
        self.voronoi_button.clicked.connect(self.voronoi_clicked)
        h.addWidget(self.voronoi_button)
        layout.addLayout(h)

        self.generate_random_points(DEFAULT_NUM_POINTS)
        self.set_diagram()

        # show the window
        self.show()

    def generate_points_clicked(self):
        print("generate points clicked")
        self.generate_random_points(int(self.num_points.text()))
        # clear the frame and reset it with the new points (conditionally)
        self.set_diagram(reset=True)

    def voronoi_clicked(self):
        print("voronoi clicked")
        self.solve_voronoi()
        self.set_diagram()

    def toggle_points_clicked(self, checked):
        print("toggle clicked", checked)
        if checked:
            self.show_points = True
        else:
            self.show_points = False
        self.set_diagram()

    # check input of the 'num points' field
    def check_input(self):
        if self.num_points.text().isdigit():
            self.points_button.setEnabled(True)
        else:
            self.points_button.setEnabled(False)

    # sets the voronoi_diagram in the frame
    # adds points to the diagram if desired
    def set_diagram(self, reset=False):
        # get a new blank canvas
        if reset:
            self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)

        # add the points to the canvas (or not)
        if self.show_points:
            image = self.voronoi_diagram.copy()
            for pt in self.points:
                image = cv2.circle(image, (pt[0], pt[1]), 2, (0, 0, 0), -1)
        else:
            image = self.voronoi_diagram

        # switch the diagram to PyQt form and put it in the image frame
        q_image = QImage(image.data, WIDTH, HEIGHT, QImage.Format_RGB888)  # .rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(q_image))

    # generate new, random points
    def generate_random_points(self, num_points):
        # generate the new points
        pt_list = []
        for _ in range(num_points):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            pt_list.append((x, y))
        self.points = pt_list
        # self.points = [(10, 10), (400, 400), (10, 400)]

    def solve_voronoi(self):
        self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)

        self.points = sorted(self.points)  # , key=lambda k: [k[1], k[0]])

        for i in tqdm(range(len(self.voronoi_diagram))):
            for j in range(len(self.voronoi_diagram[i])):
                min_dist = np.inf
                min_dist_ctr = 0
                for point in self.points:
                    # we are too far to be considered as the closest point
                    if abs(j - point[0]) > min_dist or abs(i - point[1]) > min_dist:
                        # print(min_dist)
                        # print((j, i), point)
                        continue

                    pixel = np.array((j, i))
                    np_point = np.array(point)
                    # print(pixel, point)
                    dist = round(np.linalg.norm(pixel - np_point))

                    # update the minimum distance
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_ctr = 1
                    # if two points have the same distance to the pixel, note it
                    elif dist == min_dist:
                        # canvas[i][j] = (0, 0, 0)
                        min_dist_ctr += 1
                    # the remaining points won't be closer than the min distance (since the points are sorted)
                    elif j - point[0] > min_dist:
                        break

                # this pixel is in between two points, paint it black
                if min_dist_ctr > 1:
                    self.voronoi_diagram[i][j] = (0, 0, 0)
                # if min_dist_ctr > 2:
                #     canvas = cv2.circle(canvas, (j, i), 2, (0, 255, 0), -1)
                #     print("green")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
