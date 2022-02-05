
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(root: str, batch_size: int):
    '''make DataLoaders'''
    train = ImageFolder.asloader(
        root, 'train', batch_size)
    val   = ImageFolder.asloader(
        root, 'val', batch_size, False, num_workers=1, pin_memory=False)
    test  = ImageFolder.asloader(
        root, 'test', 1, False, False, 1, False)
    return train, val, test

def default_transform(test=False):
    '''make image transformations'''
    transform = [
        T.RandomResizedCrop(224, scale=(0.3, 1.)),
        T.ToTensor(),
        # Danbooru2020 mean and std
        T.Normalize([0.7106, 0.6574, 0.6511], [0.2561, 0.2617, 0.2539])]
    if not test:
        transform.insert(1, T.RandAugment())
    return T.Compose(transform)

class ImageFolder(Dataset):
    '''dataset from a folder

    FOLDER STRUCTURE
    root - class A - 0000.jpg
                   - 0001.png
                   - ...
         - class B - ...
         - ...
    '''
    def __init__(self,
        root, split, transform=None
    ) -> None:
        super().__init__()

        self._class_names = sorted([os.path.basename(folder) for folder in glob.glob(os.path.join(root, '*'))])
        images = []
        labels = []
        for index, class_name in enumerate(self._class_names):
            temp_images = sorted(glob.glob(os.path.join(root, class_name, '**', '*'), recursive=True))
            temp_images = [file for file in temp_images if os.path.isfile(file)]
            temp_labels = [index for _ in range(len(temp_images))]
            images.extend(temp_images)
            labels.extend(temp_labels)
        train_images, valtest_images, train_labels, valtest_labels = train_test_split(
            images, labels, test_size=0.2, random_state=3407, shuffle=True)
        val_images, test_images, val_labels, test_labels = train_test_split(
            valtest_images, valtest_labels, test_size=0.5, random_state=3407, shuffle=True)

        _is_test = True
        if split == 'train':
            self.images = train_images
            self.labels = train_labels
            _is_test = False
        elif split == 'val':
            self.images = val_images
            self.labels = val_labels
        elif split == 'test':
            self.images = test_images
            self.labels = test_labels

        self.transform = transform if transform is not None else default_transform(_is_test)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        return image, self.labels[index]

    @property
    def class_names(self):
        return self._class_names

    @classmethod
    def asloader(cls, root, split,
        batch_size, shuffle=True, drop_last=True,
        num_workers=os.cpu_count(), pin_memory=torch.cuda.is_available()
    ):
        dataset = cls(root, split)
        return DataLoader(
            dataset, batch_size, drop_last=drop_last, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
