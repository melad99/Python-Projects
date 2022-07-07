'HW#2-Q2'
'Student: Melad Alsaleh'
'ID: 120170346'
import numpy as np
from scipy import ndimage
import cv2


img = np.array([[0, 0, 0, 0], [0, 20, 10, 0], [0, 30, 50, 0], [0, 0, 0, 0]], np.uint8)
img = np.pad(img,(0,0),'constant', constant_values=(0))

# 1. Arithmetic mean filter
kernel = np.ones((3,3))/9
dst = cv2.filter2D(img,-1,kernel)
print('1. Arithmetic mean filter:')
print(dst)


# 2. Geometric mean filter
rows, cols = img.shape[:2]
ksize = 3
padsize = int((ksize-1)/2)
pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
geomean = np.zeros_like(img)
for r in range(rows):
    for c in range(cols):
        geomean[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
geomean = np.uint8(geomean)

print('2. Geometric mean filter:')
print(geomean)


# 3. median filter
dst = cv2.medianBlur(img,3)
print('3. median filter:')
print(dst)

# 4. Laplacian filter
kernel = np.array([[0,-1,0], [-1, 4, -1], [0, -1, 0]])
lap = cv2.filter2D(img,-1,kernel)
print('4. Laplacian filter:')
print(lap)


# 5. roberts
roberts_cross = np.array([[-1, 0],[0, 1]])
edged_img = cv2.filter2D(img,-1,roberts_cross)
print('5. roberts operator:')
print(edged_img)

