import numpy as np
from pyflann import FLANN
from torch.utils.data import Dataset

class AdaptiveEpsilonDataset(Dataset):
    def __init__(self, X, Y, eps, batch_size):
        self.X = X
        self.Y = Y
        self.eps = eps
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, i): 
        X = self.X[i*self.batch_size:(i+1)*self.batch_size]
        Y = self.Y[i*self.batch_size:(i+1)*self.batch_size]
        eps = self.eps[i*self.batch_size:(i+1)*self.batch_size]
        print(i, self.batch_size, len(Y), len(self.Y))
        return X[0], Y[0], eps[0]

def adaptive_epsilon(loader, target_epsilon, batch_size):
    # split dataset into classes
    class_dict = dict()
    for i, (X,y) in enumerate(loader):
        y = y.item()
        X = X.numpy()
        if not y in class_dict:
            class_dict[y] = [X]
        else:
            class_dict[y].append(X)

    # build flann index for each class
    flann_dict = dict()
    for y in class_dict:
        mflann = FLANN()
        class_examples = np.array(class_dict[y])
        class_size = len(class_examples)
        image_shape = class_examples.shape[1:]
        mflann.build_index(class_examples.reshape(class_size, np.prod(image_shape)))
        flann_dict[y] = mflann

    # for each example input, find distance to the closest example input of other classes
    dataset_with_dist = []

    for i, (X,y) in enumerate(loader):
        y = y.item()
        X = X.numpy()
        smallest_dist = np.inf
        for _y in class_dict:
            if _y != y:
                _, dist = flann_dict[_y].nn_index(X.reshape(-1), 1)
                if dist[0] < smallest_dist:
                    smallest_dist = dist[0]

        dataset_with_dist.append(np.array([X,y, smallest_dist]))

    # scale the distance to [0, target_epsilon] interval
    dataset_with_eps = np.array(dataset_with_dist)
    dataset_with_eps[:,2] = target_epsilon * dataset_with_eps[:,2] / np.max(dataset_with_eps[:,2])

    # create and return dataset loader
    X = dataset_with_eps[:,0]
    Y = dataset_with_eps[:,1]
    eps = dataset_with_eps[:,2]

    return AdaptiveEpsilonDataset(X, Y, eps, batch_size)