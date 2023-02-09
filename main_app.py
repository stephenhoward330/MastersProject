import sys

import cv2
import cv2.ximgproc as ximg
import numpy as np
import random
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QDir
from time import time

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 500
DEFAULT_NUM_POINTS = 500
DEFAULT_REGION_SIZE = 16
DEFAULT_ITERATIONS = 20


# TODO: sample points on both sides of borders
# TODO: look at evenly spaced points for voronoi


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.diagram = np.full((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), 255, np.uint8)
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
        self.main_layout = QVBoxLayout()
        main_widget = QWidget()

        # Set the central widget of the Window
        main_widget.setLayout(self.main_layout)
        self.setCentralWidget(main_widget)

        # add the image frame
        self.image_frame = QLabel("Your Image Will Show Here")
        self.image_frame.setMinimumHeight(DEFAULT_HEIGHT)
        self.image_frame.setMinimumWidth(DEFAULT_WIDTH)
        self.image_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.image_frame)

        # add the diagram frame
        self.diagram_frame = QLabel()
        self.diagram_frame.setMinimumHeight(DEFAULT_HEIGHT)
        self.diagram_frame.setMinimumWidth(DEFAULT_WIDTH)
        self.diagram_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.diagram_frame)

        h = QHBoxLayout()
        # add the 'upload image' button
        self.upload_image_button = QPushButton("Upload Image")
        self.upload_image_button.clicked.connect(self.upload_image_clicked)
        h.addWidget(self.upload_image_button)
        # add the 'save image' button
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image_clicked)
        h.addWidget(self.save_image_button)

        # add the voronoi and superpixel radio buttons
        self.v_radio_button = QRadioButton("Voronoi Mode")
        self.v_radio_button.setChecked(True)
        self.v_radio_button.toggled.connect(self.voronoi_mode_clicked)
        h.addWidget(self.v_radio_button)
        self.s_radio_button = QRadioButton("Superpixel Mode")
        self.s_radio_button.toggled.connect(self.superpixel_mode_clicked)
        h.addWidget(self.s_radio_button)
        self.main_layout.addLayout(h)

        # ################ VORONOI SECTION
        self.voronoi_frame = QFrame()
        h = QHBoxLayout()
        # add the 'number of points' text and box
        h.addWidget(QLabel('Number of Points: '))
        self.num_points_field = QLineEdit(str(DEFAULT_NUM_POINTS))
        self.num_points_field.setFixedWidth(75)
        self.num_points_field.textChanged.connect(self.check_points_input)
        h.addWidget(self.num_points_field)
        # add the 'generate random points' button
        self.generate_random_points_button = QPushButton("Generate Random Points")
        self.generate_random_points_button.clicked.connect(self.generate_random_clicked)
        h.addWidget(self.generate_random_points_button)
        # add the 'generate smart points' button
        self.generate_smart_points_button = QPushButton("Generate Smart Points")
        self.generate_smart_points_button.clicked.connect(self.generate_smart_clicked)
        self.generate_smart_points_button.setEnabled(False)
        h.addWidget(self.generate_smart_points_button)

        self.voronoi_frame.setLayout(h)
        self.main_layout.addWidget(self.voronoi_frame)
        self.mode = "voronoi"

        # ################ SUPERPIXEL SECTION
        self.superpixel_frame = QFrame()
        h = QHBoxLayout()
        # add the 'region size' text and box
        h.addWidget(QLabel('Region Size: '))
        self.region_size_field = QLineEdit(str(DEFAULT_REGION_SIZE))
        self.region_size_field.textChanged.connect(self.check_superpixel_input)
        h.addWidget(self.region_size_field)
        # add the 'number of iterations' text and box
        h.addWidget(QLabel('Iterations: '))
        self.iterations_field = QLineEdit(str(DEFAULT_ITERATIONS))
        self.iterations_field.textChanged.connect(self.check_superpixel_input)
        h.addWidget(self.iterations_field)

        self.superpixel_frame.setLayout(h)
        self.main_layout.addWidget(self.superpixel_frame)
        self.superpixel_frame.hide()

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
        # add the 'generate voronoi' button
        self.generate_diagram_button = QPushButton("Generate Voronoi Diagram")
        self.generate_diagram_button.clicked.connect(self.generate_diagram_clicked)
        h.addWidget(self.generate_diagram_button)
        self.main_layout.addLayout(h)
        # layout.addLayout(v)

        # add the progress bar
        h = QHBoxLayout()
        h.addWidget(QLabel('Progress: '))
        self.progress_bar = QProgressBar()
        # self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self.progress_bar)
        self.main_layout.addLayout(h)

        self.generate_random_points(DEFAULT_NUM_POINTS)
        self.set_diagram()

        # show the window
        self.show()

    def upload_image_clicked(self) -> None:
        if self.image is None:
            image_files = None

            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setNameFilter("Image (*.png *.jpg *.jpeg)")
            dialog.setViewMode(QFileDialog.ViewMode.List)
            dialog.setDirectory(QDir.currentPath()+'/images')
            dialog.setWindowTitle("Open Image")

            if dialog.exec():
                image_files = dialog.selectedFiles()
            if image_files is None:
                print("None entered")
                return
            # if len(image_files) > 1:  # impossible with 'ExistingFile'
            #     print("Multiple files entered, using the first")
            image_file = image_files[0]
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
                self.check_points_input()
            except Exception as e:
                print("Exception during file open:", e)
                quit()
        else:
            self.image = None
            # self.im_height = DEFAULT_HEIGHT
            # self.im_width = DEFAULT_WIDTH

            self.image_frame.clear()
            self.image_frame.setText("Your Image Will Show Here")

            self.set_diagram(reset=True)
            self.upload_image_button.setText("Upload Image")
            self.generate_smart_points_button.setEnabled(False)

    def save_image_clicked(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save As", "outputs/result.png", "*.png;;*.jpg;;*.jpeg")
        # "Image (*.png *.jpg *.jpeg)")

        if filename == "":
            print("None entered")
            return
        if not (filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-5:] == '.jpeg'):
            print("BAD FILE EXTENSION")
            return

        # prepare output to save
        output = self.draw_diagram()
        output = output.astype(np.uint8)

        # save it out
        try:
            if cv2.imwrite(filename, output):
                print("file save succeeded")
            else:
                print("file save failed")
        except Exception as e:
            print("Exception during file save:", e)
            quit()

    def voronoi_mode_clicked(self, checked: bool) -> None:
        if checked:
            self.mode = "voronoi"
            self.superpixel_frame.hide()
            self.voronoi_frame.show()
            self.generate_diagram_button.setText("Generate Voronoi Diagram")
            self.generate_diagram_button.setEnabled(True)
            self.toggle_points_box.setEnabled(True)
            self.set_diagram(reset=True)

    def superpixel_mode_clicked(self, checked: bool) -> None:
        if checked:
            self.mode = "superpixel"
            self.voronoi_frame.hide()
            self.superpixel_frame.show()
            self.generate_diagram_button.setText("Generate Superpixels")
            self.check_superpixel_input()
            self.toggle_points_box.setEnabled(False)
            self.set_diagram(reset=True)

    def generate_random_clicked(self) -> None:
        self.generate_random_points(int(self.num_points_field.text()))
        # clear the frame and reset it with the new points
        # self.im_height = DEFAULT_HEIGHT
        # self.im_width = DEFAULT_WIDTH
        self.set_diagram(reset=True)

    def generate_smart_clicked(self) -> None:
        self.generate_smart_points(int(self.num_points_field.text()))
        # clear the frame and reset it with the new points
        self.set_diagram(reset=True)

    def toggle_points_clicked(self, checked: bool) -> None:
        if checked:
            self.show_points = True
        else:
            self.show_points = False
        self.set_diagram()

    def toggle_lines_clicked(self, checked: bool) -> None:
        if checked:
            self.show_lines = True
        else:
            self.show_lines = False
        self.set_diagram()

    def toggle_colors_clicked(self, checked: bool) -> None:
        if checked:
            self.show_colors = True
        else:
            self.show_colors = False
        self.set_diagram()

    def generate_diagram_clicked(self) -> None:
        if self.mode == "voronoi":
            self.generate_voronoi()
        else:
            self.generate_superpixels()

    def generate_voronoi(self) -> None:
        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        depth_map = None
        # the color map has a different integer for each area
        color_map = np.zeros((self.im_height, self.im_width), np.int32)

        # mathematical approach to generating a colored voronoi diagram with the given points
        # https://gist.github.com/bert/1188638/78a80d1824ffb2b64c736550d62b3e770e5a45b5

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

            self.progress_bar.setValue(int((i + 1) * 90 / len(self.points)))

        image_colors = np.empty((len(self.points), 3), dtype=np.uint8)

        for i in range(len(self.points)):
            if self.image is None:
                image_colors[i] = random.choices(range(256), k=3)
            else:
                # compute each cell's color as the average of all pixels in the cell
                mask = np.where(color_map == i, 255, 0)
                mask = mask.astype(np.uint8)
                mean = cv2.mean(self.image, mask)
                mean = [int(x) for x in mean[:3]]
                image_colors[i] = mean
                self.progress_bar.setValue(90 + int((i + 1) * 10 / len(self.points)))
        self.diagram = image_colors[color_map]

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
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)
        self.set_diagram()

    def generate_superpixels(self) -> None:
        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        if self.image is None:
            im = np.full((self.im_height, self.im_width, 3), 255, np.uint8)
        else:
            im = self.image.copy()
        # gaussian blur
        im = cv2.GaussianBlur(im, (5, 5), 0)
        # Convert to LAB
        src_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # convert to LAB

        # SLIC
        cv_slic = ximg.createSuperpixelSLIC(src_lab, algorithm=ximg.SLICO,
                                            region_size=int(self.region_size_field.text()))

        # for _ in range(int(self.iterations_field.text())):
        #     cv_slic.iterate(1)
        cv_slic.iterate(int(self.iterations_field.text()))

        labels = cv_slic.getLabels()
        num_regions = cv_slic.getNumberOfSuperpixels()

        image_colors = np.empty((num_regions, 3), dtype=np.uint8)

        for i in range(num_regions):
            if self.image is None:
                image_colors[i] = random.choices(range(256), k=3)
            else:
                mask = np.where(labels == i, 255, 0)
                mask = mask.astype(np.uint8)
                mean = cv2.mean(self.image, mask)
                mean = [int(x) for x in mean[:3]]
                image_colors[i] = mean
                self.progress_bar.setValue(50 + int((i + 1) * 50 / num_regions))
        self.diagram = image_colors[labels]

        # find the lines (borders between colors)
        # vertical borders
        v_lines = np.where(labels[:-1] != labels[1:], True, False)
        v_lines = np.vstack([[False] * self.im_width, v_lines])
        # horizontal borders
        h_lines = np.where(labels[:, :-1] != labels[:, 1:], True, False)
        h_lines = np.hstack([[[False]] * self.im_height, h_lines])
        # combine and save it
        self.line_diagram = h_lines | v_lines

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)
        self.set_diagram()

    # check input of the 'num points' field
    def check_points_input(self) -> None:
        if self.num_points_field.text().isdigit() and int(self.num_points_field.text()) >= 1:
            self.generate_random_points_button.setEnabled(True)
            if self.image is not None:
                self.generate_smart_points_button.setEnabled(True)
        else:
            self.generate_random_points_button.setEnabled(False)
            self.generate_smart_points_button.setEnabled(False)

    def check_superpixel_input(self) -> None:
        if self.region_size_field.text().isdigit() and int(self.region_size_field.text()) >= 1 \
                and self.iterations_field.text().isdigit() and int(self.iterations_field.text()) >= 1:
            self.generate_diagram_button.setEnabled(True)
        else:
            self.generate_diagram_button.setEnabled(False)

    # sets the voronoi_diagram in the frame
    def set_diagram(self, reset: bool = False) -> None:
        # get a new blank canvas
        if reset:
            self.diagram = np.full((self.im_height, self.im_width, 3), 255, np.uint8)
            self.line_diagram = None

        image = self.draw_diagram()

        # switch the diagram to PyQt form and put it in the image frame
        q_image = QImage(image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
        self.diagram_frame.setPixmap(QPixmap.fromImage(q_image))

    # adds points, lines, and colors to the diagram if desired
    def draw_diagram(self) -> np.ndarray:
        # get the colored diagram (or not)
        if self.show_colors:
            image = self.diagram.copy()
        else:
            image = np.full((self.im_height, self.im_width, 3), 255, np.uint8)

        # add the points to the canvas (or not)
        if self.mode == "voronoi" and self.show_points:
            if len(self.points) >= 3000:
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
            # black_img = np.zeros((500, 600, 3))
            # image = np.where(self.line_diagram is True, black_img, image)
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if self.line_diagram[i][j]:
                        image[i][j] = (0, 0, 0)

        return image

    # generate new, random points
    def generate_random_points(self, num_points: int) -> None:
        self.points = []
        while len(self.points) < num_points:
            x = random.randint(0, self.im_width - 1)
            y = random.randint(0, self.im_height - 1)
            if (x, y) in self.points:
                continue
            self.points.append((x, y))
        self.points = sorted(self.points)  # not necessary but may be nice

    # generate new, smart points
    def generate_smart_points(self, num_points: int) -> None:
        self.points = []

        # sample dark areas more
        # pixels = []
        # weights = []
        # for i in range(len(self.image)):
        #     for j in range(len(self.image[i])):
        #         pixels.append((j, i))
        #         # sample dark pixels more (a white pixel sums to 765)
        #         weights.append(768 - self.image[i][j][0] - self.image[i][j][1] - self.image[i][j][2])
        # self.points = random.choices(pixels, weights, k=num_points)

        # define the vertical filter
        vertical_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # define the horizontal filter
        horizontal_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        down_edges = cv2.filter2D(self.image, -1, vertical_filter)
        right_edges = cv2.filter2D(self.image, -1, horizontal_filter)
        up_edges = cv2.filter2D(self.image, -1, vertical_filter * -1)
        left_edges = cv2.filter2D(self.image, -1, horizontal_filter * -1)

        c_edges = cv2.addWeighted(down_edges, 1.0, right_edges, 1.0, 0.0)
        c_edges = cv2.addWeighted(c_edges, 1.0, up_edges, 1.0, 0.0)
        c_edges = cv2.addWeighted(c_edges, 1.0, left_edges, 1.0, 0.0)

        # cv2.imwrite("images/edges.png", c_edges)

        min_weight = 2
        weight_reduction_factor = 150

        pixels = []
        weights = []
        # max_c = 0
        for i in range(len(c_edges)):
            for j in range(len(c_edges[i])):
                pixels.append((j, i))
                c_sum = c_edges[i][j][0].astype(np.int32) + c_edges[i][j][1].astype(np.int32) \
                    + c_edges[i][j][2].astype(np.int32) - weight_reduction_factor
                # if c_sum > max_c:
                #     max_c = c_sum
                # c_sum = 0 if c_sum < 0 else c_sum
                weights.append(min_weight if c_sum < min_weight else c_sum + min_weight)
        self.points = random.choices(pixels, weights, k=num_points)
        # print(max_c)
        # print(type(c_sum))

        self.points = sorted(self.points)  # not necessary but may be nice

    def enable_all(self, t_f: bool) -> None:
        self.upload_image_button.setEnabled(t_f)
        self.save_image_button.setEnabled(t_f)
        self.v_radio_button.setEnabled(t_f)
        self.s_radio_button.setEnabled(t_f)

        self.num_points_field.setEnabled(t_f)
        # the next two may be disabled by check_input
        self.generate_random_points_button.setEnabled(t_f)
        self.generate_smart_points_button.setEnabled(t_f)

        self.region_size_field.setEnabled(t_f)
        self.iterations_field.setEnabled(t_f)

        if self.mode == "voronoi":
            self.toggle_points_box.setEnabled(t_f)
        self.toggle_lines_box.setEnabled(t_f)
        self.toggle_colors_box.setEnabled(t_f)
        self.generate_diagram_button.setEnabled(t_f)

        if t_f:
            self.check_points_input()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
