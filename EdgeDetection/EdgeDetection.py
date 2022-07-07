

# student: Melad Alsaleh
# ID :120170346

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

image = 't4.jpg' # original image path which the filter will be applied on
input_img = imread(image) # the array representation of the input image
[nx, ny, nz] = np.shape(input_img)  # nx: height, ny: width, nz: colors (RGB)


r_img, g_img, b_img = input_img[:, :, 0], input_img[:, :, 1], input_img[:, :, 2] # Extracting each one of the RGB components

def cnvrt_color_to_gray():
    # operation will use weights and parameters to convert the color image to grayscale
    gamma = 1.400  # a parameter
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
    grayscale_img = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma
    return grayscale_img

def sobel_filter(grayscale_img):
    # define the matrices associated with the Sobel filter
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) 
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])  
    [rows, columns] = np.shape(grayscale_img) # the shape of the input grayscale image
    sobel_filtered_img = np.zeros(shape=(rows, columns))  # initialization of the output image array -> all elements are 0

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_img[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
    return sobel_filtered_img
        

grayscale_image = cnvrt_color_to_gray()
sobel_filtered_image = sobel_filter(grayscale_image)

print('Original image:')
print(input_img)
print('\n\n')
print('Filtered image:')
print(sobel_filtered_image)

# Display the original image and the Sobel filtered image
fig = plt.figure(figsize=(10, 8))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ax1.imshow(input_img)
ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))

plt.imsave('sobel_filtered_image.png', sobel_filtered_image, cmap=plt.get_cmap('gray'))