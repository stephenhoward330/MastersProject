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

WIDTH = 600
HEIGHT = 500
DEFAULT_NUM_POINTS = 5


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
        self.show_points = True
        self.show_lines = True
        self.show_colors = True
        # TODO: allow toggling of lines and colors, and find the lines from the color voronoi

        self.points = []
        self.point_colors = []

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
        # add the 'generate line voronoi' button
        self.line_voronoi_button = QPushButton("Generate Line Voronoi")
        self.line_voronoi_button.clicked.connect(self.line_voronoi_clicked)
        h.addWidget(self.line_voronoi_button)
        # add the 'generate color voronoi' button
        self.color_voronoi_button = QPushButton("Generate Color Voronoi")
        self.color_voronoi_button.clicked.connect(self.color_voronoi_clicked)
        h.addWidget(self.color_voronoi_button)
        layout.addLayout(h)

        # add the progress bar
        h = QHBoxLayout()
        h.addWidget(QLabel('Progress: '))
        self.progress_bar = QProgressBar()
        # self.progress_bar.setAlignment(Qt.AlignCenter)
        h.addWidget(self.progress_bar)
        layout.addLayout(h)

        self.progress = 0

        self.generate_random_points(DEFAULT_NUM_POINTS)
        self.set_diagram()

        # show the window
        self.show()

    def generate_points_clicked(self):
        print("generate points clicked")
        self.generate_random_points(int(self.num_points.text()))
        # clear the frame and reset it with the new points (conditionally)
        self.set_diagram(reset=True)

    def line_voronoi_clicked(self):
        print("line voronoi clicked")
        self.my_voronoi()
        self.set_diagram()

    def color_voronoi_clicked(self):
        print("color voronoi clicked")
        self.color_voronoi()
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
        if self.num_points.text().isdigit() and int(self.num_points.text()) > 1:
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
        self.points = []
        self.point_colors = []
        for _ in range(num_points):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.points.append((x, y))

            self.point_colors.append(tuple(random.choices(range(256), k=3)))
        # self.points = [(10, 10), (400, 400), (10, 400)]

    def enable_all(self, t_f):
        self.num_points.setEnabled(t_f)
        self.points_button.setEnabled(t_f)
        self.line_voronoi_button.setEnabled(t_f)
        self.color_voronoi_button.setEnabled(t_f)
        self.toggle_points_box.setEnabled(t_f)
        if t_f:
            self.check_input()

    # my naive approach to generating just the lines of a voronoi diagram with the given points
    def my_voronoi(self):
        self.enable_all(False)
        self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)

        self.points = sorted(self.points)  # , key=lambda k: [k[1], k[0]])

        self.progress_bar.setValue(0)

        length = len(self.voronoi_diagram)
        for i in tqdm(range(length)):
            for j in range(len(self.voronoi_diagram[i])):
                min_dist = np.inf
                min_dist_ctr = 0
                for point in self.points:
                    # we are too far to be considered as the closest point
                    if abs(j - point[0]) > min_dist or abs(i - point[1]) > min_dist:
                        continue

                    pixel = np.array((j, i))
                    np_point = np.array(point)
                    dist = round(np.linalg.norm(pixel - np_point))

                    # update the minimum distance
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_ctr = 1
                    # if two points have the same distance to the pixel, note it
                    elif dist == min_dist:
                        min_dist_ctr += 1
                    # the remaining points won't be closer than the min distance (since the points are sorted)
                    elif j - point[0] > min_dist:
                        break

                # this pixel is in between two points, paint it black
                if min_dist_ctr > 1:
                    self.voronoi_diagram[i][j] = (0, 0, 0)

            self.progress_bar.setValue(int((i+1)*100/length))

        self.enable_all(True)

    # mathematical approach to generating a colored voronoi diagram with the given points
    # https://gist.github.com/bert/1188638/78a80d1824ffb2b64c736550d62b3e770e5a45b5
    def color_voronoi(self):
        self.enable_all(False)

        depth_map = None
        color_map = np.zeros((HEIGHT, WIDTH), np.int)

        self.progress_bar.setValue(0)

        def hypot(Y, X):
            return (X - x) ** 2 + (Y - y) ** 2

        for i, (x, y) in enumerate(self.points):
            # matrix with each cell representing the distance from it to the point
            paraboloid = np.fromfunction(hypot, (HEIGHT, WIDTH))
            if i == 0:
                depth_map = paraboloid.copy()
            else:
                color_map = np.where(paraboloid < depth_map, i, color_map)
                depth_map = np.where(paraboloid < depth_map, paraboloid, depth_map)
            self.progress_bar.setValue(int((i+1)*100/len(self.points)))

        self.voronoi_diagram = np.empty((HEIGHT, WIDTH, 3), np.int8)
        self.point_colors = np.array(self.point_colors)
        self.voronoi_diagram[:, :, ] = self.point_colors[color_map]

        self.enable_all(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
