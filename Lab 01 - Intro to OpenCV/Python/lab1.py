import cv2

image_path = "image.jpg"

# image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
# cv2.imshow("Picture in RGB", image_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_hsv = cv2.imread(image_path, cv2.COLOR_RGB2HSV)
# cv2.imshow("Picture in HSV", image_hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_rgb_modified = image_rgb.copy()
# image_rgb_modified[:, :, 0] = 255  # blue
# image_rgb_modified[:, :, 1] = 255  # green
# image_rgb_modified[:, :, 2] = 255  # red
# cv2.imshow("Picture modified with RGB 255, 255 255", image_rgb_modified)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)  # read default RGB (or BGR Blue Green Red - use BGR!!!)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)  # convert the downloaded RGB image to HSV
image_hsv[:, :, 2] = 255  # convert the V channel to 255
image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)  # convert the modified HSV image back to RGB
cv2.imshow("Picture in RGB", image_rgb)  # print the image
cv2.waitKey(0)
cv2.destroyAllWindows()