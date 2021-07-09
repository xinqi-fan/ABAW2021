import os.path
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
import torch

from torch.utils.data import dataset

class DatasetBase(dataset.Dataset):
    def __init__(self, opt, train_mode='Train', transform=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._transform = None
        self._train_mode = None
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

class AffWild2EXPRSequenceDataset(dataset.Dataset):
    def __init__(self, label_path, seq_len=8, train_mode='Train', transform=None):
        super(AffWild2EXPRSequenceDataset, self).__init__()
        self._name = 'EXPR'
        self._train_mode = train_mode
        self.seq_len = seq_len

        if transform is not None:
            self._transform = transform
        else:
            self._create_transform()

        # read dataset
        self._read_dataset_paths(label_path)

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        frames_ids = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row['label']
            frame_id = row['frames_ids']
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)

        # pack data
        # sample = {'image': torch.stack(images, dim=0),
        #           'label': np.array(labels),
        #           'path': img_paths,
        #           'index': index,
        #           'id': frames_ids
        #           }

        sample = {'image': torch.stack(images, dim=0),
                  'label': torch.tensor(np.array(labels)),
                  'path': img_paths,
                  'index': index,
                  'id': frames_ids
                  }
        # print (time.time() - start_time)
        # return sample

        return sample['image'], sample['label']

    def _read_dataset_paths(self, label_path):

        self._data = self._read_path_label(label_path)
        # sample them
        seq_len = self.seq_len
        self.sample_seqs = []
        if self._train_mode == 'Train':
            N = seq_len // 2
        else:
            N = seq_len
        for video in self._data.keys():
            data = self._data[video]
            for i in range(len(data) // N):
                start, end = i * N, i * N + seq_len
                if end >= len(data):
                    start, end = len(data) - seq_len, len(data)
                new_df = data.iloc[start:end]
                if not len(new_df) == seq_len:
                    assert len(new_df) < seq_len
                    count = seq_len - len(new_df)
                    for _ in range(count):
                        new_df = new_df.append(new_df.iloc[-1])
                assert len(new_df) == seq_len
                self.sample_seqs.append(new_df)
        # self.sample_seqs = self.sample_seqs[-10:] # debug only
        self._ids = np.arange(len(self.sample_seqs))
        self._dataset_size = len(self._ids)

    def __len__(self):
        return self._dataset_size

    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        data = data['EXPR_Set']
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Train_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation")

        return data

    def _create_transform(self):
        print('Using dataset own transforms')
        if self._train_mode == 'Train':
            img_size = 224
            resize = int(img_size * 1.2)
            transform_list = [transforms.Resize(resize),
                              transforms.RandomCrop(img_size),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                              ]
        else:
            img_size = 224
            transform_list = [transforms.Resize(img_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                              ]
        self._transform = transforms.Compose(transform_list)

if __name__ == '__main__':
    label_path = '/home/xinqifan2/Data/Facial_Expression/Aff-Wild2/ABAW-2021/Annotation/annotations/annotations.pkl'
    mydata = AffWild2EXPRSequenceDataset(label_path=label_path, train_mode='Train')
    mydata.__getitem__(10)
