import sys

import cv2
import numpy as np
import random
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from time import time

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

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 500
DEFAULT_NUM_POINTS = 500

# TODO: allow the user to save the result to file
# TODO: sample more points in darker regions of the image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.voronoi_diagram = np.full((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), 255, np.uint8)
        self.line_diagram = None
        self.image = None

        self.show_points = True
        self.show_lines = True
        self.show_colors = True

        self.im_height = DEFAULT_HEIGHT
        self.im_width = DEFAULT_WIDTH

        self.progress = 0

        self.points = []

        self.setWindowTitle("Voronoi Art Maker")
        layout = QVBoxLayout()
        main_widget = QWidget()

        # Set the central widget of the Window
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # add the image frame
        self.image_frame = QLabel("Your Image Will Show Here")
        self.image_frame.setMinimumHeight(DEFAULT_HEIGHT)
        self.image_frame.setMinimumWidth(DEFAULT_WIDTH)
        self.image_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_frame)

        # add the diagram frame
        self.diagram_frame = QLabel()
        layout.addWidget(self.diagram_frame)

        h = QHBoxLayout()
        # add the 'number of points' text and box
        h.addWidget(QLabel('Number of Points: '))
        self.num_points_field = QLineEdit(str(DEFAULT_NUM_POINTS))
        self.num_points_field.setFixedWidth(75)
        self.num_points_field.textChanged.connect(self.check_input)
        h.addWidget(self.num_points_field)
        self.generate_points_button = QPushButton("Generate Random Points")
        self.generate_points_button.clicked.connect(self.generate_points_clicked)
        h.addWidget(self.generate_points_button)
        # add the 'generate points' button
        self.upload_image_button = QPushButton("Upload Image")
        self.upload_image_button.clicked.connect(self.upload_image_clicked)
        h.addWidget(self.upload_image_button)
        layout.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel('Show:'))
        # add the toggle points checkbox
        self.toggle_points_box = QCheckBox('Points?')
        self.toggle_points_box.setChecked(True)
        self.toggle_points_box.stateChanged.connect(self.toggle_points_clicked)
        h.addWidget(self.toggle_points_box)
        # add the toggle lines checkbox
        self.toggle_lines_box = QCheckBox('Lines?')
        self.toggle_lines_box.setChecked(True)
        self.toggle_lines_box.stateChanged.connect(self.toggle_lines_clicked)
        h.addWidget(self.toggle_lines_box)
        # add the toggle colors checkbox
        self.toggle_colors_box = QCheckBox('Colors?')
        self.toggle_colors_box.setChecked(True)
        self.toggle_colors_box.stateChanged.connect(self.toggle_colors_clicked)
        h.addWidget(self.toggle_colors_box)
        # add the 'generate line voronoi' button
        # self.line_voronoi_button = QPushButton("Generate Line Voronoi")
        # self.line_voronoi_button.clicked.connect(self.line_voronoi_clicked)
        # h.addWidget(self.line_voronoi_button)
        # add the 'generate color voronoi' button
        self.color_voronoi_button = QPushButton("Generate Voronoi Diagram")
        self.color_voronoi_button.clicked.connect(self.color_voronoi_clicked)
        h.addWidget(self.color_voronoi_button)
        layout.addLayout(h)

        # add the progress bar
        h = QHBoxLayout()
        h.addWidget(QLabel('Progress: '))
        self.progress_bar = QProgressBar()
        # self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self.progress_bar)
        layout.addLayout(h)

        self.generate_random_points(DEFAULT_NUM_POINTS)
        self.set_diagram()

        # show the window
        self.show()

    def generate_points_clicked(self):
        print("generate points clicked")
        self.generate_random_points(int(self.num_points_field.text()))
        # clear the frame and reset it with the new points
        # self.im_height = DEFAULT_HEIGHT
        # self.im_width = DEFAULT_WIDTH
        self.set_diagram(reset=True)

    def upload_image_clicked(self):
        if self.image is None:
            print("upload image clicked")
            image_files = None

            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            dialog.setViewMode(QFileDialog.ViewMode.List)
            dialog.setWindowTitle("Open Image")

            if dialog.exec():
                image_files = dialog.selectedFiles()
            if image_files is None:
                print("None entered")
                return
            # if len(image_files) > 1:  # impossible with 'ExistingFile'
            #     print("Multiple files entered, using the first")
            image_file = image_files[0]
            print(image_file)
            try:
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                # print(image.shape)
                self.image = cv2.resize(image, (self.im_width, self.im_height))
                # self.im_height = self.image.shape[0]
                # self.im_width = self.image.shape[1]

                q_image = QImage(self.image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
                self.image_frame.setPixmap(QPixmap.fromImage(q_image))

                self.set_diagram(reset=True)
                self.upload_image_button.setText("Clear Image")
            except Exception as e:
                print(e)
                quit()
        else:
            print("clear image clicked")
            self.image = None
            # self.im_height = DEFAULT_HEIGHT
            # self.im_width = DEFAULT_WIDTH

            self.image_frame.clear()
            self.image_frame.setText("Your Image Will Show Here")

            self.set_diagram(reset=True)
            self.upload_image_button.setText("Upload Image")

    def toggle_points_clicked(self, checked):
        print("toggle points clicked", checked)
        if checked:
            self.show_points = True
        else:
            self.show_points = False
        self.set_diagram()

    def toggle_lines_clicked(self, checked):
        print("toggle lines clicked", checked)
        if checked:
            self.show_lines = True
        else:
            self.show_lines = False
        self.set_diagram()

    def toggle_colors_clicked(self, checked):
        print("toggle colors clicked", checked)
        if checked:
            self.show_colors = True
        else:
            self.show_colors = False
        self.set_diagram()

    # def line_voronoi_clicked(self):
    #     print("line voronoi clicked")
    #     self.my_voronoi()
    #     self.set_diagram()

    def color_voronoi_clicked(self):
        print("color voronoi clicked")
        self.generate_voronoi()
        self.set_diagram()

    # check input of the 'num points' field
    def check_input(self):
        if self.num_points_field.text().isdigit() and int(self.num_points_field.text()) >= 1:
            self.generate_points_button.setEnabled(True)
        else:
            self.generate_points_button.setEnabled(False)

    # sets the voronoi_diagram in the frame
    # adds points to the diagram if desired
    def set_diagram(self, reset=False):
        # get a new blank canvas
        if reset:
            self.voronoi_diagram = np.full((self.im_height, self.im_width, 3), 255, np.uint8)
            self.line_diagram = None

        if self.show_colors:
            image = self.voronoi_diagram.copy()
        else:
            image = np.full((self.im_height, self.im_width, 3), 255, np.uint8)

        # add the points to the canvas (or not)
        if self.show_points:
            if len(self.points) >= 2000:
                radius = 0
            elif len(self.points) >= 200:
                radius = 1
            elif len(self.points) >= 50:
                radius = 2
            else:
                radius = 3
            for pt in self.points:
                image = cv2.circle(image, (pt[0], pt[1]), radius, (0, 0, 0), -1)

        # add the lines to the canvas (or not)
        if self.show_lines and self.line_diagram is not None:
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if self.line_diagram[i][j]:
                        image[i][j] = (0, 0, 0)

        # switch the diagram to PyQt form and put it in the image frame
        if self.image is None:
            q_image = QImage(image.data, self.im_width, self.im_height, QImage.Format.Format_RGB888)
        else:
            q_image = QImage(image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
        self.diagram_frame.setPixmap(QPixmap.fromImage(q_image))

    # generate new, random points
    def generate_random_points(self, num_points):
        # generate the new points
        self.points = []
        while len(self.points) < num_points:
            x = random.randint(0, self.im_width-1)
            y = random.randint(0, self.im_height-1)
            if (x, y) in self.points:
                continue
            self.points.append((x, y))
        self.points = sorted(self.points)  # not necessary but may be nice

    def enable_all(self, t_f):
        self.num_points_field.setEnabled(t_f)
        self.generate_points_button.setEnabled(t_f)
        self.upload_image_button.setEnabled(t_f)
        # self.line_voronoi_button.setEnabled(t_f)
        self.color_voronoi_button.setEnabled(t_f)
        self.toggle_points_box.setEnabled(t_f)
        self.toggle_lines_box.setEnabled(t_f)
        self.toggle_colors_box.setEnabled(t_f)
        if t_f:
            self.check_input()

    # my naive approach to generating just the lines of a voronoi diagram with the given points
    # def my_voronoi(self):
    #     self.enable_all(False)
    #     self.voronoi_diagram = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    #
    #     self.points = sorted(self.points)  # , key=lambda k: [k[1], k[0]])
    #
    #     self.progress_bar.setValue(0)
    #
    #     length = len(self.voronoi_diagram)
    #     for i in tqdm(range(length)):
    #         for j in range(len(self.voronoi_diagram[i])):
    #             min_dist = np.inf
    #             min_dist_ctr = 0
    #             for point in self.points:
    #                 # we are too far to be considered as the closest point
    #                 if abs(j - point[0]) > min_dist or abs(i - point[1]) > min_dist:
    #                     continue
    #
    #                 pixel = np.array((j, i))
    #                 np_point = np.array(point)
    #                 dist = round(np.linalg.norm(pixel - np_point))
    #
    #                 # update the minimum distance
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     min_dist_ctr = 1
    #                 # if two points have the same distance to the pixel, note it
    #                 elif dist == min_dist:
    #                     min_dist_ctr += 1
    #                 # the remaining points won't be closer than the min distance (since the points are sorted)
    #                 elif j - point[0] > min_dist:
    #                     break
    #
    #             # this pixel is in between two points, paint it black
    #             if min_dist_ctr > 1:
    #                 self.voronoi_diagram[i][j] = (0, 0, 0)
    #
    #         self.progress_bar.setValue(int((i+1)*100/length))
    #
    #     self.enable_all(True)

    # mathematical approach to generating a colored voronoi diagram with the given points
    # https://gist.github.com/bert/1188638/78a80d1824ffb2b64c736550d62b3e770e5a45b5
    def generate_voronoi(self):
        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        depth_map = None
        # the color map has a different integer for each area
        color_map = np.zeros((self.im_height, self.im_width), np.int32)

        image_colors = np.empty((len(self.points), 3), dtype=np.uint8)

        def hypotenuse(Y, X):
            return (X - x) ** 2 + (Y - y) ** 2

        for i, (x, y) in enumerate(self.points):
            # matrix with each cell representing the distance from it to the point
            paraboloid = np.fromfunction(hypotenuse, (self.im_height, self.im_width))
            if i == 0:
                depth_map = paraboloid.copy()
            else:
                color_map = np.where(paraboloid < depth_map, i, color_map)
                depth_map = np.where(paraboloid < depth_map, paraboloid, depth_map)

            if self.image is None:
                image_colors[i] = random.choices(range(256), k=3)
            # else:
            #     image_colors[i] = self.image[y][x]

            self.progress_bar.setValue(int((i + 1) * 90 / len(self.points)))

        # clear the diagram
        self.voronoi_diagram = np.empty((self.im_height, self.im_width, 3), np.int8)
        if self.image is None:
            # apply random colors to each cell
            self.voronoi_diagram[:, :, ] = image_colors[color_map]
        else:
            # compute each cells color as the average of all pixels in the cell
            for i in range(len(self.points)):
                mask = np.where(color_map == i, 255, 0)
                mask = mask.astype(np.uint8)
                res = cv2.mean(self.image, mask)
                res = [int(x) for x in res[:3]]
                image_colors[i] = res
                self.progress_bar.setValue(90 + int((i + 1) * 10 / len(self.points)))
            self.voronoi_diagram[:, :, ] = image_colors[color_map]

        # find the lines (borders between colors)
        # vertical borders
        v_lines = np.where(color_map[:-1] != color_map[1:], True, False)
        v_lines = np.vstack([[False] * self.im_width, v_lines])
        # horizontal borders
        h_lines = np.where(color_map[:, :-1] != color_map[:, 1:], True, False)
        h_lines = np.hstack([[[False]] * self.im_height, h_lines])
        # combine and save it
        self.line_diagram = h_lines | v_lines

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time()-time_1, 1)) + " s")

        self.enable_all(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
