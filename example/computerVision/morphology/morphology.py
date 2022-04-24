# %% 膨胀
import cv2
import numpy as np

img = cv2.imread('./morphology.jpg',cv2.IMREAD_GRAYSCALE)

k = np.ones((7,7),dtype=np.uint8)

# erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
imgErode = cv2.erode(img,k,iterations=1)

cv2.imshow('erosion',np.hstack(( img,imgErode )))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 膨胀
import cv2
import numpy as np

img = cv2.imread('./morphology.jpg',cv2.IMREAD_GRAYSCALE)

k = np.ones((7,7),dtype=np.uint8)

# erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
imgErode = cv2.erode(img,k,iterations=1)

cv2.imshow('erosion',np.hstack(( img,imgErode )))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 膨胀
import cv2
import numpy as np

img = cv2.imread('./morphology.jpg',cv2.IMREAD_GRAYSCALE)

k = np.ones((7,7),dtype=np.uint8)

# dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
imgErode = cv2.dilate(img,k,iterations=1)


cv2.imshow('dilate',np.hstack(( img,imgErode )))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 形态学操作
import cv2
import numpy as np

img = cv2.imread('./morphology.jpg',cv2.IMREAD_GRAYSCALE)

k = np.ones((7,7),dtype=np.uint8)

# dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
imgErode = cv2.dilate(img,k,iterations=1)


cv2.imshow('erosion',np.hstack(( img,imgErode )))
cv2.waitKey(0)
cv2.destroyAllWindows()