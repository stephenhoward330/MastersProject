import os
import sys

import math
import cv2
import cv2.ximgproc as ximg
import numpy as np
import random
import typing
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QRadioButton, QButtonGroup, QCheckBox, \
    QLabel, QFrame, QFileDialog, QPushButton, QLineEdit, QProgressBar, QApplication
from PyQt6.QtCore import Qt, QDir
from pyx import canvas, text, path, bbox, color, unit
from time import time

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 500
DEFAULT_NUM_POINTS = 250
DEFAULT_REGION_SIZE = 64
DEFAULT_ITERATIONS = 20
DEFAULT_ORIENTATION = 'horizontal'

# desired physical size of the final completed diagram in inches
PDF_WIDTH = 6
PDF_HEIGHT = 5

# size of the material (each color) in inches
MATERIAL_WIDTH = 12
MATERIAL_HEIGHT = 8

# TODO: allow for images of all sizes / resolutions
# TODO: let PDF size and material size be set in the app
# Note: with color palette and no image, colors are evenly split


class Region:
    def __init__(self, center: tuple = None, corners: list = None, size: tuple = None, location: tuple = None):
        self.center = center
        self.corners = corners
        self.size = size
        self.location = location

    def sort_corners(self) -> None:
        # order the corners in each region clockwise
        def angle_between(p1, p2, c):
            x1, y1 = p1[0] - c[0], p1[1] - c[1]
            x2, y2 = p2[0] - c[0], p2[1] - c[1]
            return math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)

        self.corners = sorted(self.corners, key=lambda p: angle_between(p, (1, 0), self.center))

    def calc_size(self) -> None:
        if len(self.corners) == 0:
            self.size = 0

        max_y = -np.inf
        max_x = -np.inf
        min_y = np.inf
        min_x = np.inf

        for y, x in self.corners:
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x

        self.size = max_y - min_y, max_x - min_x
        self.location = min_y, min_x

    def calc_center(self) -> None:
        self.center = [sum(x)/len(x) for x in zip(*self.corners)]

    def scale_corners(self, y_scale, x_scale, y_offset, x_offset) -> list:
        assert self.size is not None
        corners = [(y - self.location[0], x - self.location[1]) for y, x in self.corners]
        return [(y * y_scale + y_offset, x * x_scale + x_offset) for y, x in corners]

    def scale_size(self, y_scale, x_scale) -> tuple:
        if self.size is None:
            return -1, -1
        return self.size[0] * y_scale, self.size[1] * x_scale

    def scale_center(self, y_scale, x_scale, y_offset, x_offset):
        return (self.center[0] - self.location[0]) * y_scale + y_offset, \
            (self.center[1] - self.location[1]) * x_scale + x_offset


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.diagram = np.full((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), 255, np.uint8)
        self.line_diagram = None
        self.image = None
        self.color_map = None
        self.color_palette_rgb = self.read_color_palette()  # dict of names: rgb
        self.color_palette_regions = None  # dict of names: list of regions
        self.regions = None

        self.show_points = True
        self.show_lines = True
        self.show_colors = True

        self.im_height = DEFAULT_HEIGHT
        self.im_width = DEFAULT_WIDTH

        self.progress = 0
        self.points = []
        self.mode = "voronoi"

        self.setWindowTitle("Voronoi Art Maker")
        self.main_layout = QVBoxLayout()
        self.main_widget = QWidget()

        self.frames = QVBoxLayout()

        # the frame for the user-entered image
        self.image_frame = QLabel("Your Image Will Show Here")
        self.image_frame.setMinimumHeight(DEFAULT_HEIGHT)
        self.image_frame.setMinimumWidth(DEFAULT_WIDTH)
        self.image_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # the frame for the resulting diagram
        self.diagram_frame = QLabel()
        self.diagram_frame.setMinimumHeight(DEFAULT_HEIGHT)
        self.diagram_frame.setMinimumWidth(DEFAULT_WIDTH)

        # the 'open image' button
        self.open_image_button = QPushButton("Open Image")
        self.open_image_button.clicked.connect(self.open_image_clicked)
        # the 'save image' button
        self.save_result_button = QPushButton("Save Result")
        self.save_result_button.clicked.connect(self.save_result_clicked)
        # the 'save PDFs' button
        self.save_pdfs_button = QPushButton("Save PDFs")
        self.save_pdfs_button.clicked.connect(self.save_pdfs_clicked)
        if len(self.color_palette_rgb) == 0:
            self.save_pdfs_button.setEnabled(False)

        # add the voronoi and superpixel radio buttons
        mode_group = QButtonGroup(self.main_widget)
        self.voronoi_radio_button = QRadioButton("Voronoi Mode")
        self.voronoi_radio_button.setChecked(True)
        self.voronoi_radio_button.toggled.connect(self.voronoi_mode_clicked)
        mode_group.addButton(self.voronoi_radio_button)
        self.superpixel_radio_button = QRadioButton("Superpixel Mode")
        self.superpixel_radio_button.toggled.connect(self.superpixel_mode_clicked)
        mode_group.addButton(self.superpixel_radio_button)

        # ################ VORONOI SECTION
        self.voronoi_frame = QFrame()
        # the 'number of points' text and box
        self.num_points_label = QLabel('Number of Points: ')
        self.num_points_field = QLineEdit(str(DEFAULT_NUM_POINTS))
        self.num_points_field.setFixedWidth(75)
        self.num_points_field.textChanged.connect(self.check_points_input)
        # the 'generate random points' button
        self.generate_random_points_button = QPushButton("Generate Random Points")
        self.generate_random_points_button.clicked.connect(self.generate_random_clicked)
        # the 'generate smart points' button
        self.generate_smart_points_button = QPushButton("Generate Smart Points")
        self.generate_smart_points_button.clicked.connect(self.generate_smart_clicked)
        self.generate_smart_points_button.setEnabled(False)

        # ################ SUPERPIXEL SECTION
        self.superpixel_frame = QFrame()
        # the 'region size' text and box
        self.region_size_label = QLabel('Region Size: ')
        self.region_size_field = QLineEdit(str(DEFAULT_REGION_SIZE))
        self.region_size_field.setFixedWidth(75)
        self.region_size_field.textChanged.connect(self.check_superpixel_input)
        # the 'number of iterations' text and box
        self.iterations_label = QLabel('Iterations: ')
        self.iterations_field = QLineEdit(str(DEFAULT_ITERATIONS))
        self.iterations_field.setFixedWidth(75)
        self.iterations_field.textChanged.connect(self.check_superpixel_input)
        # the 'number of superpixels' text box
        self.num_superpixels_field = QLabel('Superpixels: _____')
        self.num_superpixels_field.setFixedWidth(150)

        self.show_label = QLabel('Show:')
        # the toggle points checkbox
        self.toggle_points_box = QCheckBox('Points?')
        self.toggle_points_box.setChecked(True)
        self.toggle_points_box.stateChanged.connect(self.toggle_points_clicked)
        # the toggle lines checkbox
        self.toggle_lines_box = QCheckBox('Lines?')
        self.toggle_lines_box.setChecked(True)
        self.toggle_lines_box.stateChanged.connect(self.toggle_lines_clicked)
        # the toggle colors checkbox
        self.toggle_colors_box = QCheckBox('Colors?')
        self.toggle_colors_box.setChecked(True)
        self.toggle_colors_box.stateChanged.connect(self.toggle_colors_clicked)
        # the 'generate voronoi' button
        self.generate_diagram_button = QPushButton("Generate Voronoi Diagram")
        self.generate_diagram_button.clicked.connect(self.generate_diagram_clicked)
        self.generate_diagram_button.setFixedWidth(200)

        # the progress bar
        self.progress_label = QLabel('Progress: ')
        self.progress_bar = QProgressBar()

        # the orientation radio buttons
        orientation_group = QButtonGroup(self.main_widget)
        self.vertical_radio_button = QRadioButton("Vertical Window")
        self.vertical_radio_button.toggled.connect(self.vertical_orientation_clicked)
        orientation_group.addButton(self.vertical_radio_button)
        self.horizontal_radio_button = QRadioButton("Horizontal Window")
        self.horizontal_radio_button.toggled.connect(self.horizontal_orientation_clicked)
        orientation_group.addButton(self.horizontal_radio_button)

        if DEFAULT_ORIENTATION == 'vertical':
            self.vertical_radio_button.setChecked(True)
        else:
            self.horizontal_radio_button.setChecked(True)

        # initialize the points and diagram
        self.generate_random_points(DEFAULT_NUM_POINTS)
        self.set_diagram()

        # order the layout
        self.init_layout(DEFAULT_ORIENTATION)

        # show the window
        self.show()

    def init_layout(self, orientation: str) -> None:
        # clear the main layout
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Set the central widget of the Window
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        if orientation == 'horizontal':
            self.frames = QHBoxLayout()
        else:
            self.frames = QVBoxLayout()

        # add the image frame
        self.frames.addWidget(self.image_frame)
        # add the diagram frame
        self.frames.addWidget(self.diagram_frame)
        self.main_layout.addLayout(self.frames)

        h = QHBoxLayout()
        # add the 'open image' button
        h.addWidget(self.open_image_button)
        # add the 'save image' button
        h.addWidget(self.save_result_button)
        # add the 'save PDFs' button
        h.addWidget(self.save_pdfs_button)
        # add the voronoi and superpixel radio buttons
        h.addWidget(self.voronoi_radio_button)
        h.addWidget(self.superpixel_radio_button)
        self.main_layout.addLayout(h)

        # ################ VORONOI SECTION
        self.voronoi_frame = QFrame()
        h = QHBoxLayout()
        # add the 'number of points' text and box
        h.addWidget(self.num_points_label)
        h.addWidget(self.num_points_field)
        # add the 'generate random points' button
        h.addWidget(self.generate_random_points_button)
        # add the 'generate smart points' button
        h.addWidget(self.generate_smart_points_button)

        self.voronoi_frame.setLayout(h)
        self.main_layout.addWidget(self.voronoi_frame)

        # ################ SUPERPIXEL SECTION
        self.superpixel_frame = QFrame()
        h = QHBoxLayout()
        # add the 'region size' text and box
        h.addWidget(self.region_size_label)
        h.addWidget(self.region_size_field)
        # add the 'number of iterations' text and box
        h.addWidget(self.iterations_label)
        h.addWidget(self.iterations_field)
        # add the 'number of superpixels' text box
        h.addWidget(self.num_superpixels_field)

        self.superpixel_frame.setLayout(h)
        self.main_layout.addWidget(self.superpixel_frame)

        # show the correct fields for the current mode
        if self.mode == 'voronoi':
            self.superpixel_frame.hide()
            self.voronoi_frame.show()
        else:
            self.voronoi_frame.hide()
            self.superpixel_frame.show()

        h = QHBoxLayout()
        # add the toggle points, lines, and colors checkboxes
        h.addWidget(self.show_label)
        h.addWidget(self.toggle_points_box)
        h.addWidget(self.toggle_lines_box)
        h.addWidget(self.toggle_colors_box)
        # add the 'generate voronoi' button
        h.addWidget(self.generate_diagram_button)
        self.main_layout.addLayout(h)

        # add the progress bar
        h = QHBoxLayout()
        h.addWidget(self.progress_label)
        h.addWidget(self.progress_bar)
        # add the orientation radio buttons
        h.addWidget(self.vertical_radio_button)
        h.addWidget(self.horizontal_radio_button)
        self.main_layout.addLayout(h)

        # resize the window
        self.setFixedSize(self.main_layout.sizeHint())

    @staticmethod
    def read_color_palette() -> typing.Dict[str, tuple]:
        # reads palette.txt
        if not os.path.exists("palette.txt"):
            return {}

        palette = {}
        with open("palette.txt", 'r') as f:
            for line in f:
                if line[0] == '#' or line[0] == '\n':
                    continue

                name, line = line.split(' ', 1)
                line = line.replace('(', '').replace(')', '')  # remove parentheses
                color_tup = tuple(map(int, line.split(',')))  # form tuple

                if len(color_tup) != 3:
                    raise Exception("Invalid palette.txt file: colors should have 3 channels (RGB), one color per line")
                if min(color_tup) < 0 or max(color_tup) > 255:
                    raise Exception("Invalid palette.txt file: RGB values should be between 0 and 255")

                palette[name] = color_tup[::-1]  # reverse the tuple into BGR form

        return palette

    def reset_palette_regions(self) -> None:
        self.color_palette_regions = {}
        for name in self.color_palette_rgb.keys():
            self.color_palette_regions[name] = []

    def open_image_clicked(self) -> None:
        if self.image is None:
            # open image
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
                return
            image_file = image_files[0]

            try:
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                if image is None:
                    raise Exception("invalid file path")
                self.image = cv2.resize(image, (self.im_width, self.im_height))
                # self.im_height = self.image.shape[0]
                # self.im_width = self.image.shape[1]

                q_image = QImage(self.image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
                self.image_frame.setPixmap(QPixmap.fromImage(q_image))

                self.set_diagram(reset=True)
                self.open_image_button.setText("Clear Image")
                self.check_points_input()
            except Exception as e:
                print("Exception during file open:", e)
                quit()
        else:
            # clear image
            self.image = None
            # self.im_height = DEFAULT_HEIGHT
            # self.im_width = DEFAULT_WIDTH

            self.image_frame.clear()
            self.image_frame.setText("Your Image Will Show Here")

            self.set_diagram(reset=True)
            self.open_image_button.setText("Open Image")
            self.generate_smart_points_button.setEnabled(False)

    def save_result_clicked(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save As", "outputs/result.png", "*.png;;*.jpg;;*.jpeg")
        # "Image (*.png *.jpg *.jpeg)")

        if filename == "":
            return
        if filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-5:] == '.jpeg':
            # prepare output to save as image
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
        else:
            print("BAD FILE EXTENSION")
            print("Should be .png, .jpg, or .jpeg")

    def save_pdfs_clicked(self) -> None:
        # saves out all files that a laser cutter would need to the 'pdfs' folder
        if self.line_diagram is None or len(self.color_palette_rgb) == 0:
            print("NO LINE DIAGRAM AND/OR COLOR PALETTE")
            return

        if self.mode != 'voronoi' and self.image is not None:
            print("Warning: Superpixels with image, may give poor results")

        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        # for resizing to desired pdf size
        x_scale = PDF_WIDTH / self.im_width
        y_scale = PDF_HEIGHT / self.im_height

        # clear pdfs folder
        for filename in os.listdir("pdfs"):
            file_path = os.path.join("pdfs", filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        # find all the regions
        self.find_regions()
        self.progress_bar.setValue(20)

        for r in self.regions:
            r.calc_size()
            r.calc_center()
            r.sort_corners()
        self.progress_bar.setValue(40)

        # create the reference image (to help the user put it back together again)
        full = self.draw_diagram(True)
        text_font = cv2.FONT_HERSHEY_PLAIN
        text_scale = 1
        text_thickness = 0
        for i in range(len(self.regions)):
            text_num = str(i + 1)
            text_size, _ = cv2.getTextSize(text_num, text_font, text_scale, text_thickness)
            text_origin = (round(self.regions[i].center[1] - text_size[0] // 2),
                           round(self.regions[i].center[0] + text_size[1] // 2))

            cv2.putText(full, text_num, text_origin, text_font, text_scale, (0, 0, 0), text_thickness)
        full = cv2.resize(full, (PDF_WIDTH*100, PDF_HEIGHT*100))
        cv2.imwrite("pdfs/Reference.png", full)

        # set units to inches
        unit.set(defaultunit="inch")
        engine = text.UnicodeEngine(size=PDF_WIDTH*2)  # results in a good size / adjustable

        self.progress_bar.setValue(50)

        margin_size = 1/4  # room around the edge (inches)
        padding_size = 1/8  # room between pieces (inches)

        for j, name in enumerate(self.color_palette_rgb.keys()):  # for each specified color...
            if len(self.color_palette_regions[name]) == 0:
                # in this case, the color doesn't appear, skip it
                continue

            x_offset = margin_size
            y_offset = margin_size
            row_height = 0

            c = canvas.canvas()
            for contour_number in self.color_palette_regions[name]:  # for each region...

                region_size = self.regions[contour_number].scale_size(y_scale, x_scale)
                # see if we need to move to the next line
                if x_offset + region_size[1] > MATERIAL_WIDTH - margin_size:
                    x_offset = margin_size
                    y_offset += row_height + padding_size
                    row_height = 0
                # check if we have enough material
                if y_offset + region_size[0] > MATERIAL_HEIGHT - margin_size:
                    print("Not enough material:", name)
                    self.progress_bar.setValue(100)
                    self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")
                    self.enable_all(True)
                    return

                contour = self.regions[contour_number].scale_corners(y_scale, x_scale, y_offset, x_offset)
                for i in range(len(contour)):  # step through the points of the region (contour)
                    if i == 0:
                        old_y, old_x = contour[i]
                        continue
                    y, x = contour[i]
                    if i == 1:
                        p = path.line(old_x, old_y, x, y)
                        old_x = x
                        old_y = y
                    elif i == len(contour) - 1:  # at the end of the path, close it
                        p = p << path.line(old_x, old_y, x, y)
                        p.append(path.closepath())
                    else:
                        p = p << path.line(old_x, old_y, x, y)
                        old_x = x
                        old_y = y

                # draw the path on the canvas
                c.stroke(p, [color.rgb.red])  # style.linewidth(0.1),
                # put number in region
                center = self.regions[contour_number].scale_center(y_scale, x_scale, y_offset, x_offset)
                c.insert(engine.text(center[1], center[0], str(contour_number+1), [text.halign.center]))

                # update offsets
                if region_size[0] > row_height:
                    row_height = region_size[0]
                x_offset += region_size[1] + padding_size

            c.writePDFfile(f"pdfs/{name}.pdf", page_bbox=bbox.bbox(0, 0, MATERIAL_WIDTH, MATERIAL_HEIGHT))

            self.progress_bar.setValue(50 + int((j + 1) * 50 / len(self.color_palette_rgb)))

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    def find_regions(self) -> None:
        # finds each region of the diagram, so that they can be separated for printing/cutting
        assert self.color_map is not None

        regions = [[] for _ in range(self.color_map.max()+1)]  # list of lists, list i has the corners of region i
        margin_map = self.color_map.copy()
        margin_map = np.pad(margin_map, 5, constant_values=-1)

        # find all the corners
        for i in range(len(margin_map)):
            if i < 4 or i > len(margin_map)-5:  # in a margin area
                continue
            for j in range(len(margin_map[i])):
                if j < 4 or j > len(margin_map[i]) - 5:  # in a margin area
                    continue

                # look at a 2x2 window
                window_set = set()
                window_set.add(margin_map[i][j])
                window_set.add(margin_map[i+1][j])
                window_set.add(margin_map[i][j+1])
                window_set.add(margin_map[i+1][j+1])

                if len(window_set) > 2:  # more than 2 regions represented in the window = corner
                    y = i
                    x = j

                    # adjust to ensure point is in bounds
                    if i <= 4:
                        y += 1
                    if j <= 4:
                        x += 1

                    pt = (y-5, x-5)
                    for num in window_set:
                        if num == -1:
                            continue
                        regions[num].append(pt)

        # add corners at corners of image
        regions[self.color_map[0, 0]].append((0, 0))
        regions[self.color_map[self.im_height-1, 0]].append((self.im_height-1, 0))
        regions[self.color_map[0, self.im_width-1]].append((0, self.im_width-1))
        regions[self.color_map[self.im_height-1, self.im_width-1]].append((self.im_height-1, self.im_width-1))

        # save as a list of Region objects
        self.regions = []
        for i in range(len(regions)):
            self.regions.append(Region(corners=regions[i]))

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
        # clear the frame and reset it with new points
        self.generate_random_points(int(self.num_points_field.text()))
        # self.im_height = DEFAULT_HEIGHT
        # self.im_width = DEFAULT_WIDTH
        self.set_diagram(reset=True)

    def generate_smart_clicked(self) -> None:
        # clear the frame and reset it with new points
        self.generate_smart_points(int(self.num_points_field.text()))
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

        self.reset_palette_regions()

        depth_map = None
        # the color map has a different integer for each area
        self.color_map = np.zeros((self.im_height, self.im_width), np.int32)

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
                # used to color the regions, first region all 0's, second region all 1's and so on
                self.color_map = np.where(paraboloid < depth_map, i, self.color_map)
                # used for determining which point is closest
                depth_map = np.where(paraboloid < depth_map, paraboloid, depth_map)

            self.progress_bar.setValue(int((i + 1) * 90 / len(self.points)))

        self.color_and_find_lines(len(self.points), 90)

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)
        self.set_diagram()

    def generate_superpixels(self) -> None:
        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        self.reset_palette_regions()

        if self.image is None:
            im = np.full((self.im_height, self.im_width, 3), 255, np.uint8)
        else:
            im = self.image.copy()

        # gaussian blur
        im = cv2.GaussianBlur(im, (5, 5), 0)
        # Convert to LAB format
        src_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # convert to LAB

        # SLIC
        cv_slic = ximg.createSuperpixelSLIC(src_lab, algorithm=ximg.SLICO,
                                            region_size=int(self.region_size_field.text()))

        cv_slic.iterate(int(self.iterations_field.text()))

        self.color_map = cv_slic.getLabels()
        num_regions = cv_slic.getNumberOfSuperpixels()

        self.color_and_find_lines(num_regions, 50)

        self.num_superpixels_field.setText('Superpixels: ' + str(num_regions))

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)
        self.set_diagram()

    def color_and_find_lines(self, num_regions: int, progress_so_far: int) -> None:
        # after finding the color map through Voronoi or superpixels:
        #   determine colors for each region
        #   and find the lines between the regions

        image_colors = np.empty((num_regions, 3), dtype=np.uint8)

        # assign a color to each region
        for i in range(num_regions):
            if self.image is None:
                if len(self.color_palette_rgb) == 0:
                    # if there is no color palette, choose any random color
                    image_colors[i] = random.choices(range(256), k=3)
                else:
                    # there is a color palette, but no image
                    # random_color_name = random.choice(list(self.color_palette_rgb.keys()))
                    # pseudo-random, will balance the colors over the color palette
                    random_color_name = list(self.color_palette_rgb.keys())[i % len(self.color_palette_rgb)]
                    image_colors[i] = self.color_palette_rgb[random_color_name]
                    self.color_palette_regions[random_color_name].append(i)
            else:
                # find the average color across the region in the reference image
                mask = np.where(self.color_map == i, 255, 0)
                mask = mask.astype(np.uint8)
                mean = cv2.mean(self.image, mask)
                mean = [int(x) for x in mean[:3]]
                if len(self.color_palette_rgb) == 0:
                    # if there is no color palette, set the color to the average of the region
                    image_colors[i] = mean
                else:
                    # otherwise, round this average to the closest available color
                    if i == 0:
                        colors = np.array(list(self.color_palette_rgb.values()))
                        names = list(self.color_palette_rgb.keys())
                    mean = np.array(mean)
                    distances = np.sqrt(np.sum((colors - mean) ** 2, axis=1))
                    index_of_smallest = int(np.where(distances == np.amin(distances))[0][0])
                    image_colors[i] = colors[index_of_smallest]
                    self.color_palette_regions[names[index_of_smallest]].append(i)

                self.progress_bar.setValue(progress_so_far + int((i + 1) * (100-progress_so_far) / num_regions))
        self.diagram = image_colors[self.color_map]

        # find the lines (borders between regions)
        # vertical borders
        v_lines = np.where(self.color_map[:-1] != self.color_map[1:], True, False)
        v_lines = np.vstack([[False] * self.im_width, v_lines])
        # horizontal borders
        h_lines = np.where(self.color_map[:, :-1] != self.color_map[:, 1:], True, False)
        h_lines = np.hstack([[[False]] * self.im_height, h_lines])
        # combine and save it
        self.line_diagram = h_lines | v_lines

    def vertical_orientation_clicked(self, checked: bool) -> None:
        if checked:
            self.init_layout('vertical')

    def horizontal_orientation_clicked(self, checked: bool) -> None:
        if checked:
            self.init_layout('horizontal')

    def check_points_input(self) -> None:
        # check input of the 'num points' field
        if self.num_points_field.text().isdigit() and int(self.num_points_field.text()) >= 1:
            self.generate_random_points_button.setEnabled(True)
            if self.image is not None:
                self.generate_smart_points_button.setEnabled(True)
        else:
            self.generate_random_points_button.setEnabled(False)
            self.generate_smart_points_button.setEnabled(False)

    def check_superpixel_input(self) -> None:
        # check input of the 'region size' field as well as the 'iterations' field
        if self.region_size_field.text().isdigit() and int(self.region_size_field.text()) >= 1 \
                and self.iterations_field.text().isdigit() and int(self.iterations_field.text()) >= 1:
            self.generate_diagram_button.setEnabled(True)
        else:
            self.generate_diagram_button.setEnabled(False)

    def set_diagram(self, reset: bool = False) -> None:
        # sets the diagram in the frame

        if reset:
            # get a new blank canvas
            self.diagram = np.full((self.im_height, self.im_width, 3), 255, np.uint8)
            self.line_diagram = None
            self.num_superpixels_field.setText("Superpixels: _____")

        image = self.draw_diagram()

        # switch the diagram to PyQt form and put it in the image frame
        q_image = QImage(image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
        self.diagram_frame.setPixmap(QPixmap.fromImage(q_image))

    def draw_diagram(self, all_on: bool = False) -> np.ndarray:
        # adds points, lines, and colors to the diagram if desired

        # get the colored diagram (or not)
        if self.show_colors or all_on:
            image = self.diagram.copy()
        else:
            image = np.full((self.im_height, self.im_width, 3), 255, np.uint8)

        # add the points to the canvas (or not)
        if not all_on and self.mode == "voronoi" and self.show_points:
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
        if all_on or self.show_lines and self.line_diagram is not None:
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if self.line_diagram[i][j]:
                        image[i][j] = (0, 0, 0)

        return image

    def generate_random_points(self, num_points: int) -> None:
        # generate new, random points

        if num_points > self.im_width * self.im_height / 10:
            print("TOO MANY POINTS, REQUEST LESS THAN", round(self.im_width * self.im_height / 10))
            return

        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        self.points = []
        while len(self.points) < num_points:
            x = random.randint(0, self.im_width - 1)
            y = random.randint(0, self.im_height - 1)
            if (x, y) in self.points:
                # duplicate point, skip
                continue
            self.points.append((x, y))
            self.progress_bar.setValue(int(len(self.points) * 100 / num_points))

        self.points = sorted(self.points)  # not necessary but may be nice

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    def generate_smart_points(self, num_points: int) -> None:
        # generate new, smart points

        if num_points > self.im_width * self.im_height / 10:
            print("TOO MANY POINTS, REQUEST LESS THAN", round(self.im_width * self.im_height / 10))
            return

        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        self.points = []

        # ############# EDGE DETECTION

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

        min_weight = 3
        weight_reduction_factor = 150

        # ############# CALCULATE PIXEL WEIGHTS

        pixels = []
        weights = []
        for i in range(len(c_edges)):
            for j in range(len(c_edges[i])):
                pixels.append((j, i))
                c_sum = c_edges[i][j][0].astype(np.int32) + c_edges[i][j][1].astype(np.int32) \
                    + c_edges[i][j][2].astype(np.int32) - weight_reduction_factor
                weights.append(min_weight if c_sum < min_weight else c_sum + min_weight)
            self.progress_bar.setValue(int((i + 1) * 100 / len(c_edges)))

        # ##### THIS FINDS POINTS WITHOUT REPLACEMENT, BUT MUCH SLOWER
        # self.points = []
        # while len(self.points) < num_points:
        #     point = random.choices(pixels, weights, k=1)[0]
        #     if point in self.points:
        #         continue
        #     self.points.append(point)
        #     self.progress_bar.setValue(50 + int((len(self.points)) * 50 / num_points))

        # ##### MUCH FASTER, BUT WITH REPLACEMENT
        self.points = random.choices(pixels, weights, k=num_points)

        self.points = sorted(self.points)  # not necessary but may be nice

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    def enable_all(self, t_f: bool) -> None:
        # enable (or disable) all fields/buttons

        self.open_image_button.setEnabled(t_f)
        self.save_result_button.setEnabled(t_f)
        if len(self.color_palette_rgb) > 0:
            self.save_pdfs_button.setEnabled(t_f)
        self.voronoi_radio_button.setEnabled(t_f)
        self.superpixel_radio_button.setEnabled(t_f)

        self.num_points_field.setEnabled(t_f)
        # the next two may be disabled by check_input
        self.generate_random_points_button.setEnabled(t_f)
        if t_f and self.image is not None:
            self.generate_smart_points_button.setEnabled(t_f)
        elif not t_f:
            self.generate_smart_points_button.setEnabled(False)

        self.region_size_field.setEnabled(t_f)
        self.iterations_field.setEnabled(t_f)

        if self.mode == "voronoi":
            self.toggle_points_box.setEnabled(t_f)
        self.toggle_lines_box.setEnabled(t_f)
        self.toggle_colors_box.setEnabled(t_f)
        self.generate_diagram_button.setEnabled(t_f)

        self.vertical_radio_button.setEnabled(t_f)
        self.horizontal_radio_button.setEnabled(t_f)

        if t_f:
            self.check_points_input()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
