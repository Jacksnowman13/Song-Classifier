from numpy import ndarray
from mygrad import sliding_window_view
from ImageLoader import load_image
import random

def slice_image_px(x_pixels: int, image:ndarray, y_pixels=None, ):
    image = image.transpose(2, 0, 1)
    col_width = x_pixels
    if(y_pixels is not None):
        row_height = y_pixels
    else:
        row_height = x_pixels
    image = sliding_window_view(image, window_shape=(col_width, row_height), step=(col_width, row_height))
    image = image.transpose(0, 1, 3, 4, 2)
    return image.reshape((image.shape[0] * image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
def slice_image(rows: int, cols: int, image: ndarray):
    # image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
    image = image.transpose(2, 0, 1)
    col_width = int(image.shape[1]/cols)
    row_height = int(image.shape[2]/rows)
    image = sliding_window_view(image, window_shape=(col_width, row_height), step=(col_width, row_height))
    image = image.transpose(0, 1, 3, 4, 2)
    return image.reshape((image.shape[0] * image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
def get_random_slice(min_size: int, max_size:int, image):
    if max_size > image.shape[0]:
        max_size = image.shape[0]
    if max_size > image.shape[1]:
        max_size = image.shape[1]
    x_size = random.randint(min_size, max_size)
    y_size = random.randint(min_size, max_size)
    x_start = random.randint(0, image.shape[0] - x_size)
    y_start = random.randint(0, image.shape[1] - y_size)
    return image[x_start:x_start + x_size,y_start:y_start + y_size]
# img = load_image("HubbleImages\heic0109a.tif")
# img = slice_image(10, 10, img)
# print(img.shape)