import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from ImageSlicer import get_random_slice
import torch
from torch import tensor
import random

def get_dataset(entry_num: int, maps, device, image_size=500, normalize=True):
    while(entry_num % len(maps) != 0):
        entry_num -= 1
    images = torch.empty(entry_num, 3, image_size, image_size).type(torch.cuda.FloatTensor)
    labels = torch.empty(entry_num)#.type(torch.cuda.IntTensor)
    for i in range(entry_num//len(maps)):
        for j in range(len(maps)):
            ten = tensor(get_random_slice(image_size, image_size, random.choice(maps[j])))
            ten = torch.transpose(torch.transpose(ten, 0, 2), 1, 2)[None,...]
            images[i * len(maps) + j] = ten
            labels[i * len(maps) + j] = j
    
    print("Done")
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    print("Done")
    X_train, Y_train, X_test, Y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    print("Done")
    
    if normalize:
        mean_image = X_train.mean(axis=0)
        std_image = X_train.std(axis=0)

        X_train -= mean_image
        X_train /= std_image

        X_test -= mean_image
        X_test /= std_image
        
    print(device)
    train_data = X_train.to(device)
    test_data = Y_train.to(device).to(torch.int64)

    X = X_test.to(device)
    Y = Y_test.to(device).to(torch.int64)
    del(X_train, Y_train, X_test, Y_test)
    return train_data, test_data, X, Y