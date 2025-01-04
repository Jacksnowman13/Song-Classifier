# import skimage.io as io
import imagecodecs
import os
from torch import tensor
import torch
import numpy as np
# dtype = torch.cuda.FloatTensor 
def load_image(filepath: str, device='cpu'):
    # shape-(Height, Width, Color)
    # if(filepath[-3:] != "tif"):
    #     image = io.imread(str(filepath))
    # else:
    image = imagecodecs.imread(str(filepath))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    return image
def filenames_from_dict(names_and_numbers: dict, folder="images"):
    out = []
    for name in names_and_numbers.keys():
            for i in range(1, names_and_numbers[name] + 1):
                out.append(f"{folder}/{name}_{i}.jpg")
    return out
def get_filepaths(path: str):
    return [path + "/" + out for out in os.listdir(path)]
def save_npys(paths: "list[str]", filepath):
    for i in range(len(paths)):
        img = load_image(paths[i])
        if(len(img.shape) == 3 and img.mean() <= 253):
            np.save(f"{filepath}_{i}.npy", img)
        else:
            img = ((img.astype('float32')/65535) * 255).astype('uint8')
            np.save(f"{filepath}_{i}.npy", img)
def filter_bad_images(paths: "list[str]", filepath):
    for i in range(len(paths)):
        img = load_image(paths[i])
        if(len(img.shape) == 3 and img.mean() <= 253):
            np.save(f"{filepath}_{i}.npy", img)
        else:
            print(img.shape, img.mean, i)
def load_mmaps(filepath: str):
    maps = []
    i=0
    try:
        while True:
            maps.append(np.load(f"{filepath}_{i}.npy", mmap_mode='r'))
            i += 1
    except:
        return maps
# print(load_image("TelescopeClassifier\HubbleImages\heic0108a.tif").shape)
