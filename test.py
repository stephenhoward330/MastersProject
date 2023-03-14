import cv2
import cv2.ximgproc as ximg
import numpy as np
from pyx import canvas, text, path

if __name__ == '__main__':
    # # SLIC # #
    # src = cv2.imread('images/nick.jpg')  # read image
    #
    # src = cv2.resize(src, (600, 500))
    # # gaussian blur
    # blurred = cv2.GaussianBlur(src, (5, 5), 0)
    # # Convert to LAB
    # src_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)  # convert to LAB
    #
    # # SLIC
    # cv_slic = ximg.createSuperpixelSLIC(src_lab, algorithm=ximg.SLICO, region_size=8)
    # cv_slic.iterate(10)
    # labels = cv_slic.getLabels()
    # num_regions = cv_slic.getNumberOfSuperpixels()
    #
    # image_colors = np.empty((num_regions, 3), dtype=np.uint8)
    #
    # for i in range(num_regions):
    #     mask = np.where(labels == i, 255, 0)
    #     mask = mask.astype(np.uint8)
    #     mean = cv2.mean(src, mask)
    #     mean = [int(x) for x in mean[:3]]
    #     image_colors[i] = mean
    #
    # result = image_colors[labels]
    #
    # cv2.imwrite("outputs/nick_slic.png", result)
    # print("Num superpixels:", num_regions)

    c = canvas.canvas()
    # text.set(text.UnicodeEngine)
    # c.text(0, 0, "Hello, world!")
    unicode_engine = text.UnicodeEngine()
    c.insert(unicode_engine.text(0, 0, "1"))
    c.stroke(path.line(0, 0, 2, 0))
    c.writePDFfile("outputs/test.pdf")
