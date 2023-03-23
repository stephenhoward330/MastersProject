import os
import sys

import cv2
import cv2.ximgproc as ximg
import numpy as np
import random
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QRadioButton, QButtonGroup, QCheckBox, \
    QLabel, QFrame, QFileDialog, QPushButton, QLineEdit, QProgressBar, QApplication
from PyQt6.QtCore import Qt, QDir
from pyx import canvas, text, path, bbox, style, color
from time import time

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 500
DEFAULT_NUM_POINTS = 500
DEFAULT_REGION_SIZE = 64
DEFAULT_ITERATIONS = 20
DEFAULT_ORIENTATION = 'horizontal'

# TODO: allow for images of all sizes / resolutions
# TODO: determine new color palette
# TODO: move pieces together to save space
# TODO: specify size of the thing (8" x 12")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.diagram = np.full((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), 255, np.uint8)
        self.line_diagram = None
        self.image = None
        self.color_map = None
        self.color_palette = self.read_color_palette()

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
    def read_color_palette() -> dict[str, tuple]:
        if not os.path.exists("palette.txt"):
            return {}

        palette = {}
        with open("palette.txt", 'r') as f:
            for line in f:
                if line[0] == '#' or line[0] == '\n':
                    continue

                name, line = line.split(' ', 1)
                line = line.replace('(', '').replace(')', '')  # remove parens
                color_tup = tuple(map(int, line.split(',')))  # form tuple

                if len(color_tup) != 3:
                    raise Exception("Invalid palette.txt file: colors should have 3 channels (RGB), one color per line")
                if min(color_tup) < 0 or max(color_tup) > 255:
                    raise Exception("Invalid palette.txt file: RGB values should be between 0 and 255")

                # palette.append(color_tup[::-1])  # reverse the tuple into BGR form
                palette[name] = color_tup[::-1]  # reverse the tuple into BGR form

        return palette

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
                # print("None entered")
                return
            # if len(image_files) > 1:  # impossible with 'ExistingFile'
            #     print("Multiple files entered, using the first")
            image_file = image_files[0]
            try:
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                if image is None:
                    raise Exception("invalid filepath")
                # print(image.shape)
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
            # print("None entered")
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
        elif filename[-4:] == '.svg':
            # convert image to .svg format (for a laser cutter)
            # https://stackoverflow.com/questions/43108751/convert-contour-paths-to-svg-paths
            if self.line_diagram is not None:
                # im_gray = cv2.cvtColor(self.line_diagram, cv2.COLOR_BGR2GRAY)
                # _, thresh = cv2.threshold(im_gray, 127, 255, 0)
                thresh = np.where(self.line_diagram, 255, 0)
                thresh = thresh.astype('uint8')
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                with open(filename, "w+") as f:
                    f.write(f'<svg width="{self.im_width}" height="{self.im_height}" xmlns="http://www.w3.org/2000/svg">')
                    for c in contours:
                        f.write('<path d="M')
                        for i in range(len(c)):
                            x, y = c[i][0]
                            f.write(f"{x} {y} ")
                        f.write('" fill="none" stroke="white"/>')
                    f.write("</svg>")
            else:
                print("NO LINE DIAGRAM")
        else:
            print("BAD FILE EXTENSION")

    def save_pdfs_clicked(self) -> None:
        if self.line_diagram is None or len(self.color_palette) == 0:
            print("NO LINE DIAGRAM AND/OR PALETTE")
            return

        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        centers = self.find_centers()
        full = self.draw_diagram(True)

        # create the reference image (to help put it back together again)
        text_font = cv2.FONT_HERSHEY_PLAIN
        text_scale = 1
        text_thickness = 0
        for i in range(1, centers.max() + 1):
            center = np.where(centers == i)
            text_num = str(i)
            text_size, _ = cv2.getTextSize(text_num, text_font, text_scale, text_thickness)
            text_origin = (center[1][0] - text_size[0] // 2, center[0][0] + text_size[1] // 2)

            cv2.putText(full, text_num, text_origin, text_font, text_scale, (0, 0, 0), text_thickness)
        cv2.imwrite("pdfs/Reference.png", full)

        unicode_engine = text.UnicodeEngine(size=300)
        image = self.diagram.copy()

        self.progress_bar.setValue(20)

        # draw lines between cells (so neighboring cells of the same color don't combine
        for i in range(len(image)):
            for j in range(len(image[i])):
                if self.line_diagram[i][j]:
                    image[i][j] = (4, 5, 6)  # dummy value

        self.progress_bar.setValue(30)

        for j, (name, rgb_color) in enumerate(self.color_palette.items()):  # for each specified color...
            # find areas with that color
            region = np.all(image == rgb_color, axis=-1)
            region = np.where(region, 255, 0)

            if np.max(region) == 0:  # in this case, the color doesn't appear, skip it
                # print('Skipped', name)
                continue

            region = region.astype('uint8')
            contours, _ = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # this code will remove some redundant points from the contours, smoothing lines
            reduced_contours = []
            for contour in contours:
                new_contour = []
                for i in range(len(contour)):
                    if i == 0:
                        new_contour.append(contour[i][0])
                        continue
                    elif i == len(contour) - 1:
                        new_contour.append(contour[i][0])
                        break
                    a_x, a_y = contour[i-1][0]
                    b_x, b_y = contour[i][0]
                    c_x, c_y = contour[i+1][0]
                    if not self.is_between(a_x, a_y, b_x, b_y, c_x, c_y):
                        new_contour.append(contour[i][0])
                reduced_contours.append(new_contour)

            c = canvas.canvas()
            for contour in reduced_contours:  # for each region...
                for i in range(len(contour)):  # step through the points of the region
                    if i == 0:
                        old_x, old_y = contour[i]
                        contour_number = self.color_map[old_y+3][old_x+3]  # find the corresponding color map region
                        continue
                    x, y = contour[i]
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
                    # c.insert(unicode_engine.text(x, y, str(i)))

                if len(contour) > 0:
                    # draw the path on the canvas
                    c.stroke(p, [style.linewidth(0.1), color.rgb.red])
                    # put number in region
                    number_loc = np.where(centers == contour_number+1)  # printed numbers are 1-based, not 0-based
                    c.insert(unicode_engine.text(number_loc[1][0], number_loc[0][0], str(contour_number+1),
                                                 [text.halign.center]))

            if len(contours) > 0:
                c.writePDFfile(f"pdfs/{name}.pdf", page_bbox=bbox.bbox(0, 0, self.im_width, self.im_height))

            self.progress_bar.setValue(30 + int((j + 1) * 70 / len(self.color_palette)))

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    @staticmethod
    def is_between(a_x, a_y, b_x, b_y, c_x, c_y) -> bool:
        # helper method for smoothing lines in save_pdfs_clicked()
        cross_product = (c_y - a_y) * (b_x - a_x) - (c_x - a_x) * (b_y - a_y)

        # compare versus epsilon for floating point values, or != 0 if using integers
        if abs(cross_product) > 0.5:
            return False
        return True

    def find_centers(self) -> np.ndarray:
        # finds centers of each region in the color map
        centers = np.zeros((self.im_height, self.im_width), np.int32)

        for i in range(self.color_map.max() + 1):
            w = np.where(self.color_map == i)
            center = (w[1].min() + ((w[1].max() - w[1].min()) // 2), w[0].min() + ((w[0].max() - w[0].min()) // 2))

            centers[center[1]][center[0]] = i + 1

        return centers

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

        image_colors = np.empty((len(self.points), 3), dtype=np.uint8)

        for i in range(len(self.points)):
            if self.image is None:
                if len(self.color_palette) == 0:
                    image_colors[i] = random.choices(range(256), k=3)
                else:
                    image_colors[i] = random.choice(list(self.color_palette.values()))
            else:
                # compute each cell's color as the average of all pixels in the cell
                mask = np.where(self.color_map == i, 255, 0)
                mask = mask.astype(np.uint8)
                mean = cv2.mean(self.image, mask)
                mean = [int(x) for x in mean[:3]]  # removing the added 4th dimension and cast to int
                if len(self.color_palette) == 0:
                    # if there is no color palette, set the color to the average of the region
                    image_colors[i] = mean
                else:
                    # otherwise, round this average to the closest available color
                    if i == 0:
                        colors = np.array(list(self.color_palette.values()))
                    mean = np.array(mean)
                    distances = np.sqrt(np.sum((colors - mean) ** 2, axis=1))
                    index_of_smallest = int(np.where(distances == np.amin(distances))[0][0])
                    image_colors[i] = colors[index_of_smallest]
                self.progress_bar.setValue(90 + int((i + 1) * 10 / len(self.points)))
        self.diagram = image_colors[self.color_map]

        # find the lines (borders between colors)
        # vertical borders
        v_lines = np.where(self.color_map[:-1] != self.color_map[1:], True, False)
        v_lines = np.vstack([[False] * self.im_width, v_lines])
        # horizontal borders
        h_lines = np.where(self.color_map[:, :-1] != self.color_map[:, 1:], True, False)
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

        self.color_map = cv_slic.getLabels()
        num_regions = cv_slic.getNumberOfSuperpixels()

        image_colors = np.empty((num_regions, 3), dtype=np.uint8)

        for i in range(num_regions):
            if self.image is None:
                if len(self.color_palette) == 0:
                    # if there is no color palette, choose any color
                    image_colors[i] = random.choices(range(256), k=3)
                else:
                    image_colors[i] = random.choice(list(self.color_palette.values()))
            else:
                mask = np.where(self.color_map == i, 255, 0)
                mask = mask.astype(np.uint8)
                mean = cv2.mean(self.image, mask)
                mean = [int(x) for x in mean[:3]]
                if len(self.color_palette) == 0:
                    # if there is no color palette, set the color to the average of the region
                    image_colors[i] = mean
                else:
                    # otherwise, round this average to the closest available color
                    if i == 0:
                        colors = np.array(list(self.color_palette.values()))
                    mean = np.array(mean)
                    distances = np.sqrt(np.sum((colors - mean) ** 2, axis=1))
                    index_of_smallest = int(np.where(distances == np.amin(distances))[0][0])
                    image_colors[i] = colors[index_of_smallest]

                self.progress_bar.setValue(50 + int((i + 1) * 50 / num_regions))
        self.diagram = image_colors[self.color_map]

        # find the lines (borders between colors)
        # vertical borders
        v_lines = np.where(self.color_map[:-1] != self.color_map[1:], True, False)
        v_lines = np.vstack([[False] * self.im_width, v_lines])
        # horizontal borders
        h_lines = np.where(self.color_map[:, :-1] != self.color_map[:, 1:], True, False)
        h_lines = np.hstack([[[False]] * self.im_height, h_lines])
        # combine and save it
        self.line_diagram = h_lines | v_lines

        self.num_superpixels_field.setText('Superpixels: ' + str(num_regions))

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)
        self.set_diagram()

    def vertical_orientation_clicked(self, checked: bool) -> None:
        if checked:
            self.init_layout('vertical')

    def horizontal_orientation_clicked(self, checked: bool) -> None:
        if checked:
            self.init_layout('horizontal')

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
            self.num_superpixels_field.setText("Superpixels: _____")

        image = self.draw_diagram()

        # switch the diagram to PyQt form and put it in the image frame
        q_image = QImage(image.data, self.im_width, self.im_height, QImage.Format.Format_BGR888)
        self.diagram_frame.setPixmap(QPixmap.fromImage(q_image))

    # adds points, lines, and colors to the diagram if desired
    def draw_diagram(self, all_on: bool = False) -> np.ndarray:
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
            # black_img = np.zeros((500, 600, 3))
            # image = np.where(self.line_diagram is True, black_img, image)
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if self.line_diagram[i][j]:
                        image[i][j] = (0, 0, 0)

        return image

    # generate new, random points
    def generate_random_points(self, num_points: int) -> None:
        self.enable_all(False)

        time_1 = time()

        self.progress_bar.resetFormat()
        self.progress_bar.setValue(0)

        self.points = []
        while len(self.points) < num_points:
            x = random.randint(0, self.im_width - 1)
            y = random.randint(0, self.im_height - 1)
            if (x, y) in self.points:
                continue
            self.points.append((x, y))
            self.progress_bar.setValue(int(len(self.points) * 100 / num_points))
        self.points = sorted(self.points)  # not necessary but may be nice

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    # generate new, smart points
    def generate_smart_points(self, num_points: int) -> None:
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

        # cv2.imwrite("images/edges.png", c_edges)

        min_weight = 3
        weight_reduction_factor = 150

        pixels = []
        weights = []
        for i in range(len(c_edges)):
            for j in range(len(c_edges[i])):
                pixels.append((j, i))
                c_sum = c_edges[i][j][0].astype(np.int32) + c_edges[i][j][1].astype(np.int32) \
                    + c_edges[i][j][2].astype(np.int32) - weight_reduction_factor
                weights.append(min_weight if c_sum < min_weight else c_sum + min_weight)
            self.progress_bar.setValue(int((i + 1) * 100 / len(c_edges)))
        # TODO: change the below to sample without replacement
        self.points = random.choices(pixels, weights, k=num_points)

        self.points = sorted(self.points)  # not necessary but may be nice

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(str(round(time() - time_1, 1)) + " s")

        self.enable_all(True)

    def enable_all(self, t_f: bool) -> None:
        self.open_image_button.setEnabled(t_f)
        self.save_result_button.setEnabled(t_f)
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
