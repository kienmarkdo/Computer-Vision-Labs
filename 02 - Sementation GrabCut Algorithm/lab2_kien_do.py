# Nom: Kien Do
# Numéro étudiant: 300163370

# Lab #2: Segmentation d'instances (personnes)

# Dans ce deuxième laboratoire, vous avez l'opportunité de tester une fonction OpenCV: le GrabCut.
# On vous donne deux images, chacune contenant trois personnes.
# Créer une petite application Python qui utilisera le GrabCut afin d'extraire ces trois
# personnes en éliminant l'arrière-plan. Évidemment, pour ce faire, vous devrez manuellement
# masquer certains pixels (i.e. coder en dur) des images de facon à ce que le GrabCut donne un bon résultat.
# Montrer les masques utilisés; le plus simple seront ceux-ci, le meilleur est votre solution.

# Voici une référence pour vous; votre résultat final devrait ressembler à celui montrer ici:

# https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Leave one uncommented and the other one commented
# image_num = 1  # COMMENT/UNCOMMENT THIS
image_num = 2  # COMMENT/UNCOMMENT THIS

# Load the image
img = cv.imread(f"image{image_num}.png")
assert img is not None, "file could not be read, check with os.path.exists()"

# Define three rectangles, one for each person in the image (x, y, width, height)
if image_num == 1:  # first image
    rectangles = [
        (30, 100, 130, 300),
        (330, 100, 129, 300),
        (600, 110, 100, 400),
    ]
elif image_num == 2:  # second image
    rectangles = [
        (100, 100, 100, 400),
        (320, 4, 100, 300),
        (510, 190, 150, 400),
    ]


# Initialize the mask
mask = np.zeros(img.shape[:2], np.uint8)
for rect in rectangles:
    x, y, w, h = rect
    mask[y : y + h, x : x + w] = cv.GC_PR_FGD

# Perform GrabCut with the rectangles
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

# --------------------------------------------------------------------
# # Extract the segmented image
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
# segmented_img = img * mask2[:, :, np.newaxis]

# plt.imshow(segmented_img), plt.colorbar(), plt.show()
# --------------------------------------------------------------------

# Mask code
# newmask is the mask image I manually labelled
newmask = cv.imread(f"image{image_num}_mask.png", cv.IMREAD_GRAYSCALE)
assert newmask is not None, "file could not be read, check with os.path.exists()"

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(
    img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK
)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img * mask[:, :, np.newaxis]
plt.imshow(img), plt.colorbar(), plt.show()


# ------------------------------------------------------------------------------------
# Code from: https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# img = cv.imread("image2.png")
# assert img is not None, "file could not be read, check with os.path.exists()"
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (0, 0, 700, 500)
# cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
# img = img * mask2[:, :, np.newaxis]
# plt.imshow(img), plt.colorbar(), plt.show()

# # newmask is the mask image I manually labelled
# newmask = cv.imread("newmask.png", cv.IMREAD_GRAYSCALE)
# assert newmask is not None, "file could not be read, check with os.path.exists()"
# # wherever it is marked white (sure foreground), change mask=1
# # wherever it is marked black (sure background), change mask=0
# mask[newmask == 0] = 0
# mask[newmask == 255] = 1
# mask, bgdModel, fgdModel = cv.grabCut(
#     img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK
# )
# mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
# img = img * mask[:, :, np.newaxis]
# plt.imshow(img), plt.colorbar(), plt.show()
