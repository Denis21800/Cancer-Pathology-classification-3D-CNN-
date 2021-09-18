from pathlib import Path
import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def augmentation(image, out_size=(448, 448)):
    if np.random.randint(0, 10) >= 3:
        return image
    shifted_img = np.zeros(shape=image.shape, dtype=np.float)
    size_y, size_x, _ = image.shape
    pos_y0 = np.random.randint(0, size_y - out_size[0])
    pos_x0 = np.random.randint(0, size_x - out_size[1])
    pos_y1 = pos_y0 + out_size[0]
    pos_x1 = pos_x0 + out_size[1]
    crop_img = image[pos_y0:pos_y1, pos_x0:pos_x1, :]
    shifted_img[pos_y0:pos_y1, pos_x0:pos_x1, :] = crop_img.copy()
    return shifted_img


class ModelDataset(Dataset):
    def __init__(self, data, image_folder, is_val=False):
        self.data = data
        self.load_dir = image_folder
        self.is_val = is_val
        # erase_pixels = np.zeros((1, 1, 3)).astype(np.uint8)
        # erase_pixels = cv2.applyColorMap(erase_pixels, cv2.COLORMAP_JET)
        # erase_pixels = cv2.cvtColor(erase_pixels, cv2.COLOR_BGR2RGB)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        item = self.data.get(index)
        rec = item.get('data')
        label = rec.get('label')
        file_ = rec.get('file')
        image_seq = self.load_image_seq(file_, label)
        img_seq_t = [self.transforms(np.array(img)) for img in image_seq]
        o_index = rec.get('o_index')
        o_index = o_index if o_index is not None else []
        return img_seq_t, label, file_, o_index

    def load_image_seq(self, file, label):
        load_dir = Path(file)
        seq_files = load_dir.glob('*.png')
        image_seq = []
        for f in seq_files:
            image_data = cv2.imread(str(f))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB).astype(np.float)
            image_data /= 255.0
            if not self.is_val:
                image_data = augmentation(np.array(image_data))
            image_seq.append(np.array(image_data))

        return image_seq

    def __len__(self):
        return len(self.data)


class ModelData(object):
    def __init__(self, data, image_folder):
        assert data
        self.data = data
        self.train_loader = None
        self.test_loader = None
        self.all_data_loader = None
        self.val_loader = None
        self.image_folder = image_folder

    def create_model_data(self):
        train_data = {}
        test_data = {}
        val_data = {}
        test_index = 0
        train_index = 0
        val_index = 0
        for key in self.data:
            item = self.data.get(key)
            rec = item.get('data')
            is_test = rec.get('is_test')
            if is_test == 1:
                test_data.update({test_index: item})
                test_index += 1
            elif is_test == 0:
                train_data.update({train_index: item})
                train_index += 1
            elif is_test == 2:
                val_data.update({val_index: item})
                val_index += 1
        test_dataset = ModelDataset(test_data, is_val=True, image_folder=self.image_folder)
        train_dataset = ModelDataset(train_data, image_folder=self.image_folder)
        val_dataset = ModelDataset(val_data, is_val=True, image_folder=self.image_folder)
        all_data = ModelDataset(self.data, image_folder=self.image_folder)
        if train_dataset:
            self.train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=1, batch_size=1)
        if test_dataset:
            self.test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=1)
        self.all_data_loader = DataLoader(dataset=all_data, shuffle=True)
        if val_dataset:
            self.val_loader = DataLoader(dataset=val_dataset, shuffle=True)
