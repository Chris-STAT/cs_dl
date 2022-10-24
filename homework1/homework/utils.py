from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here.
        """
        #raise NotImplementedError('SuperTuxDataset.__init__')
        import csv
        import os
        import numpy as np
        image_to_tensor = transforms.ToTensor()
        self.images = []
        self.labels = []
        csv_reader = csv.reader(open(os.path.join(dataset_path,'labels.csv')))
        next(csv_reader)
        for row in csv_reader:
            self.images.append(image_to_tensor(Image.open(os.path.join(dataset_path, row[0]))))
            self.labels.append(LABEL_NAMES.index(row[1]))


    def __len__(self):
        """
        Your code here
        """
        #raise NotImplementedError('SuperTuxDataset.__len__')
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        #raise NotImplementedError('SuperTuxDataset.__getitem__')
        return self.images[idx], self.labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
