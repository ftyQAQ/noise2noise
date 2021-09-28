from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
import dxchange

def nomal(image0):

    image0 = image0.astype(np.float32)
    # x_mean = np.mean(image0)
    # x_std = np.std(image0)
    # image0 = (image0 - x_mean) / np.maximum(x_std, 1e-7)
    x_max = np.max(image0)
    x_min = np.min(image0)
    image0 = (image0-x_min)/(x_max-x_min)*255.
    h, w= image0.shape

    # h, input_channel_num=3, out_ch=3w = image0.shape
    image = np.zeros((h, w,1))
    image[:,:,0] = image0
    # image[:, :, 1] = image0
    # image[:, :, 2] = image0
    # image[:, :, :2] = image0


    return image

class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
        self.image_paths1 = [p for p in Path(image_dir+'1').glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_paths2 = [p for p in Path(image_dir+'2').glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_paths = image_dir
        self.image_num = len(self.image_paths1)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path1 = random.choice(self.image_paths1)
            image_name = str(image_path1).split('\\')[-1]
            # image_path2 = random.choice(self.image_paths2)
            image1 = dxchange.read_tiff(str(image_path1))
            image2 = dxchange.read_tiff(self.image_paths+'2/'+image_name)
            # print(image_path1,self.image_paths+'2/'+image_name)

            image1 = nomal(image1)
            image2 = nomal(image2)
            h, w, _ = image2.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image2.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                # print(h, w, i, j, i + image_size, j + image_size)
                x[sample_id] = image1[i:i + image_size, j:j + image_size]

                y[sample_id] = image2[i:i + image_size, j:j + image_size]
                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []
        self.image_dir = image_dir

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            x = dxchange.read_tiff(str(image_path))
            image_name = str(image_path).split('\\')[-1]
            y = dxchange.read_tiff(self.image_dir+'_or/'+image_name)
            x = nomal(x)
            y = nomal(y)
            h, w, _ = y.shape
            x = x[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            y = y[:(h // 16) * 16, :(w // 16) * 16]
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]


# if __name__ == '__main__':
#     image_dir = 'G:/noise2noise/data_tiff/ss'
#     generator = NoisyImageGenerator(image_dir, batch_size=1, image_size=256)
#     print()
