import numpy as np 

x = np.array([[1,3,6,8],[9,8,8,2],[5,4,2,3],[6,6,3,3]], np.int32)
def dft_matrix(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    t = np.zeros((rows,cols),complex)
    output_img = np.zeros((rows,cols),complex)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows,1))
    y = n.reshape((cols,1))
    for row in range(0,rows):
        M1 = 1j*np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
        t[row] = np.dot(M1, input_img[row])
    for col in range(0,cols):
        M2 = 1j*np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
        output_img[:,col] = np.dot(M2, t[:,col])
    return output_img

dft_img = dft_matrix(x)
print(dft_img)

h = np.array([[1, -0.2-0.2j, 0, -0.2+0.2j],
              [-0.2-0.2j, 0.05j, 0, 0.05],
              [0, 0, 0, 0],
              [-0.2-0.2j, 0.05, 0, -0.05j]])

from scipy.signal import convolve2d
def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

img = multi_convolver(dft_img, h, 2)

print(img)

noise = np.random.normal(0,0.1,img.shape)
y = img + noise
print(y)

def idftmtx(N):
    n = np.asmatrix(np.arange(N))
    return np.exp((2j * np.pi / N) * n.T * n)

def get_idft(data):
    """Get inverse discrete fourier transform of data(numpy matrix/array)."""
    if len(data.shape) == 1:
        return (idftmtx(len(data)) * data[:, None] / len(data)).A1
    M, N = data.shape
    return np.asarray(idftmtx(M) * data * idftmtx(N) / (M * N))

x_hat = get_idft(y)
Error = x_hat-x
