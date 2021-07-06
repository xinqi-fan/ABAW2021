import pandas as pd
from PIL import Image
import os
import numpy as np
import sys
import glob

import torch
from torch.utils.data import dataset
from torchvision import datasets


class AffWild2EXPRDataset(dataset.Dataset):

    def __init__(self, root_dir, img_relative_dir, label_relative_dir, data_mode='', phase='', transform=None, sequence_len=9, generate_label=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data_mode = data_mode

        self.EMOTIONS = {0: "Neutral", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Sadness", 6: "Surprise"}
        self.EMOTIONS2Index = {"Neutral": 0, "Anger": 1, "Disgust": 2, "Fear": 3, "Happiness": 4, "Sadness": 5, "Surprise": 6}

        self.img_relative_dir = img_relative_dir
        self.img_dir = os.path.join(root_dir, self.img_relative_dir)
        self.label_relative_dir = label_relative_dir

        self.seq_len = sequence_len
        self.seq_mid_len = round(self.seq_len // 2)

        if generate_label:
            if phase == 'train':
                set_relative_dir = os.path.join(self.label_relative_dir, 'EXPR_Set/Train_Set')
                label_path = os.path.join(root_dir, set_relative_dir)
                self._gen_label(self.img_dir, label_path)
            elif phase == 'validation':
                set_relative_dir = os.path.join(self.label_relative_dir, 'EXPR_Set/Validation_Set')
                label_path = os.path.join(root_dir, set_relative_dir)
                self._gen_label(self.img_dir, label_path)
            else: # test
                raise NotImplemented('Test set not implemented yet')

        if phase == 'train':
            set_relative_dir = os.path.join(self.label_relative_dir, 'EXPR_Set/Train_Set.csv')
            label_path = os.path.join(root_dir, set_relative_dir)
            self.label_frame = pd.read_csv(label_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(label_path, sep=',', header=0).iloc[:1000]    # for debug
        elif phase == 'validation':
            set_relative_dir = os.path.join(self.label_relative_dir, 'EXPR_Set/Validation_Set.csv')
            label_path = os.path.join(root_dir, set_relative_dir)
            self.label_frame = pd.read_csv(label_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(label_path, sep=',', header=0).iloc[:1000]    # for debug
        else: # test
            raise NotImplemented('Test set not implemented yet')

        # split video and image names
        # self.label_frame = self.label_frame.reset_index(drop=True)          # for debug
        video_with_img = self.label_frame.iloc[:, 0]
        self.video_img = video_with_img.str.split('/', expand=True)

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.data_mode == 'static':
            img_name = os.path.join(self.img_dir, self.label_frame.iloc[idx, 0])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)

        elif self.data_mode == 'sequence_naive':
            # image sequence index
            if idx - self.seq_mid_len < 0:
                idx_seq = list(range(idx, idx + self.seq_len))
            elif idx + self.seq_mid_len + 1 > len(self.label_frame):
                idx_seq = list(range(idx - self.seq_len + 1, idx + 1))
            else:
                idx_seq = list(range(idx - self.seq_mid_len, idx + self.seq_mid_len + 1))

            # image sequence
            img_seq = []
            img_name_seq = []
            for idx_cur in idx_seq:
                img_name = os.path.join(self.img_dir, self.label_frame.iloc[idx_cur, 0])
                img = Image.open(img_name)
                if self.transform:
                    img = self.transform(img)
                img_seq.append(img)
                img_name_seq.append(img_name)
            image = torch.stack(img_seq, dim=0)

        elif self.data_mode == 'sequence_video_middle':
            # video by video data loading
            cur_video_name = self.video_img.iloc[idx, 0]
            video_indices = self.video_img.index[self.video_img.iloc[:, 0] == cur_video_name].to_list()
            video_len = len(video_indices)
            start_idx, end_idx = video_indices[0], video_indices[-1]
            dist_to_start = idx - start_idx
            dist_to_end = end_idx - idx

            if video_len < self.seq_len:
                idx_seq = list(range(start_idx, end_idx+1))
                n_more_frame = self.seq_len - video_len
                for _ in range(n_more_frame):
                    idx_seq.append(idx)         # repeat current frame
                idx_seq.sort()
            else:
                # fix idx in the middle
                if dist_to_start < self.seq_mid_len:
                    idx_seq = list(range(idx - dist_to_start, idx + self.seq_mid_len + 1))
                    repeat = [idx - dist_to_start] * (self.seq_mid_len - dist_to_start)
                    idx_seq.extend(repeat)
                    idx_seq = sorted(idx_seq)
                elif dist_to_end < self.seq_mid_len:
                    idx_seq = list(range(idx - self.seq_mid_len, idx + dist_to_end + 1))
                    repeat = [idx + dist_to_end] * (self.seq_mid_len - dist_to_end)
                    idx_seq.extend(repeat)
                    idx_seq = sorted(idx_seq)
                else:
                    idx_seq = list(range(idx - self.seq_mid_len, idx + self.seq_mid_len + 1))
            assert len(idx_seq) == self.seq_len

            # image sequence
            img_seq = []
            img_path_seq = []
            for idx_cur in idx_seq:
                img_path = os.path.join(self.img_dir, self.label_frame.iloc[idx_cur, 0])
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                img_seq.append(img)
                img_path_seq.append(img_path)
            image = torch.stack(img_seq, dim=0)

        elif self.data_mode == 'sequence_video_middle_repeat':
            # video by video data loading
            cur_video_name = self.video_img.iloc[idx, 0]
            video_indices = self.video_img.index[self.video_img.iloc[:, 0] == cur_video_name].to_list()
            video_len = len(video_indices)
            start_idx, end_idx = video_indices[0], video_indices[-1]
            dist_to_start = idx - start_idx
            dist_to_end = end_idx - idx

            if video_len < self.seq_len:
                idx_seq = list(range(start_idx, end_idx+1))
                n_more_frame = self.seq_len - video_len
                for _ in range(n_more_frame):
                    idx_seq.append(idx)         # repeat current frame
                idx_seq.sort()
            else:
                # fix idx in the middle
                if dist_to_start < self.seq_mid_len:
                    idx_seq = list(range(idx - dist_to_start, idx + self.seq_mid_len + 1))
                    repeat = [idx] * (self.seq_mid_len - dist_to_start)
                    idx_seq.extend(repeat)
                    idx_seq = sorted(idx_seq)

                elif dist_to_end < self.seq_mid_len:
                    idx_seq = list(range(idx - self.seq_mid_len, idx + dist_to_end + 1))
                    repeat = [idx] * (self.seq_mid_len - dist_to_end)
                    idx_seq.extend(repeat)
                    idx_seq = sorted(idx_seq)
                else:
                    idx_seq = list(range(idx - self.seq_mid_len, idx + self.seq_mid_len + 1))
            assert len(idx_seq) == self.seq_len

            # image sequence
            img_seq = []
            img_path_seq = []
            for idx_cur in idx_seq:
                img_path = os.path.join(self.img_dir, self.label_frame.iloc[idx_cur, 0])
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                img_seq.append(img)
                img_path_seq.append(img_path)
            image = torch.stack(img_seq, dim=0)

        elif self.data_mode == 'sequence_video_non_middle':
            # video by video data loading
            cur_video_name = self.video_img.iloc[idx, 0]
            video_indices = self.video_img.index[self.video_img.iloc[:, 0] == cur_video_name].to_list()
            video_len = len(video_indices)
            start_idx, end_idx = video_indices[0], video_indices[-1]
            dist_to_start = idx - start_idx
            dist_to_end = end_idx - idx

            if video_len < self.seq_len:
                idx_seq = list(range(start_idx, end_idx + 1))
                n_more_frame = self.seq_len - video_len
                for _ in range(n_more_frame):
                    idx_seq.append(idx)  # repeat current frame
                idx_seq.sort()
            else:
                # not fix idx in the middle
                if dist_to_start < self.seq_mid_len:
                    last_frame_idx = idx + self.seq_mid_len + (self.seq_mid_len - dist_to_start)
                    idx_seq = list(range(last_frame_idx-self.seq_len+1, last_frame_idx+1))
                elif dist_to_end < self.seq_mid_len:
                    first_frame_idx = idx - self.seq_mid_len - (self.seq_mid_len - dist_to_end)
                    idx_seq = list(range(first_frame_idx, first_frame_idx+self.seq_len))
                else:
                    idx_seq = list(range(idx - self.seq_mid_len, idx + self.seq_mid_len + 1))
            assert len(idx_seq) == self.seq_len

            # image sequence
            img_seq = []
            img_path_seq = []
            for idx_cur in idx_seq:
                img_path = os.path.join(self.img_dir, self.label_frame.iloc[idx_cur, 0])
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                img_seq.append(img)
                img_path_seq.append(img_path)
            image = torch.stack(img_seq, dim=0)

        else:
            raise ValueError('data mode not supported: {}'.format(self.data_mode))

        # label
        labels = self.label_frame.iloc[idx, 1].astype('float')
        labels = torch.tensor(labels).long()

        return image, labels

    def _gen_label(self, img_dir, label_dir):
        label_frame = np.zeros((0, 2))
        video_dir = os.listdir(label_dir)

        for filename in video_dir:
            video_folder_name = filename[:-4]
            print(f'processing video {video_folder_name}')
            img_file_name_paths = sorted(glob.glob(os.path.join(img_dir, video_folder_name, '*.jpg')))
            img_label_path = os.path.join(label_dir, filename)

            file_data = read_process_file('EXPR', img_file_name_paths, img_label_path)
            label_frame = np.append(label_frame, file_data.label_frame, axis=0)

        label_frame = pd.DataFrame(label_frame, columns=['path', 'EXPR'])
        label_task_dir = '/'.join(label_dir.split('/')[:-1])
        set_flag = label_dir.split('/')[-1]
        label_framw_save_path = os.path.join(label_task_dir, set_flag+'.csv')
        label_frame.to_csv(label_framw_save_path, index=False)
        print(f'label saved to {label_framw_save_path}')


class AffWild2AUDataset(dataset.Dataset):

    def __init__(self, root_dir, phase, transform=None, generate_label=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'Cropped_aligned_image/cropped_aligned')
        self.transform = transform

        if generate_label:
            if phase == 'train':
                label_relative_dir = 'Annotation/annotations/AU_Set/Train_Set'
                label_path = os.path.join(root_dir, label_relative_dir)
                self._gen_label(self.img_dir, label_path)
            elif phase == 'validation':
                label_relative_dir_FER = 'Annotation/annotations/AU_Set/Validation_Set'
                label_path = os.path.join(root_dir, label_relative_dir_FER)
                self._gen_label(self.img_dir, label_path)
            else: # test
                raise NotImplemented('Test set not implemented yet')

        if phase == 'train':
            anno_relative_path = 'Annotation/annotations/AU_Set/Train_Set.csv'
            anno_path = os.path.join(root_dir, anno_relative_path)
            self.label_frame = pd.read_csv(anno_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(anno_path, sep=',', header=0).iloc[:1000]    # for debug
        elif phase == 'validation':
            anno_relative_path = 'Annotation/annotations/AU_Set/Validation_Set.csv'
            anno_path = os.path.join(root_dir, anno_relative_path)
            self.label_frame = pd.read_csv(anno_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(anno_path, sep=',', header=0).iloc[:1000]    # for debug
        else: # test
            raise NotImplemented('Test set not implemented yet')

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.label_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        labels = self.label_frame.iloc[idx, 1:].astype('float')
        labels = torch.tensor(labels)

        return image, labels

    def _gen_label(self, img_dir, label_dir):

        label_frame = np.zeros((0, 13))
        video_dir = os.listdir(label_dir)

        for filename in video_dir:
            video_folder_name = filename[:-4]
            print(f'processing video {video_folder_name}')
            img_file_name_paths = sorted(glob.glob(os.path.join(img_dir, video_folder_name, '*.jpg')))
            img_label_path = os.path.join(label_dir, filename)

            file_data = read_process_file('AU', img_file_name_paths, img_label_path)
            label_frame = np.append(label_frame, file_data.label_frame, axis=0)

        label_frame = pd.DataFrame(label_frame, columns=['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15',
                                            'AU23', 'AU24', 'AU25', 'AU26'])
        label_task_dir = '/'.join(label_dir.split('/')[:-1])
        set_flag = label_dir.split('/')[-1]
        label_framw_save_path = os.path.join(label_task_dir, set_flag+'.csv')
        label_frame.to_csv(label_framw_save_path, index=False)
        print(f'label saved to {label_framw_save_path}')


class AffWild2ExprAuDataset(dataset.Dataset):

    def __init__(self, root_dir, phase, transform=None, generate_label=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'Cropped_aligned_image/cropped_aligned')
        self.label_dir = os.path.join(root_dir, 'Annotation/annotations/')

        if generate_label:
            if phase == 'train':
                label_relative_dir_FER = 'Annotation/annotations/EXPR_Set/Train_Set'
                label_path_FER = os.path.join(root_dir, label_relative_dir_FER)
                label_relative_dir_AU = 'Annotation/annotations/AU_Set/Train_Set'
                label_path_AU = os.path.join(root_dir, label_relative_dir_AU)
                self._gen_matched_label(self.img_dir, label_path_FER, label_path_AU)
            elif phase == 'validation':
                label_relative_dir_FER = 'Annotation/annotations/EXPR_Set/Validation_Set'
                label_path_FER = os.path.join(root_dir, label_relative_dir_FER)
                label_relative_dir_AU = 'Annotation/annotations/AU_Set/Validation_Set'
                label_path_AU = os.path.join(root_dir, label_relative_dir_AU)
                self._gen_matched_label(self.img_dir, label_path_FER, label_path_AU)
            else: # test
                raise NotImplemented('Test set not implemented yet')

        if phase == 'train':
            anno_relative_path = 'Annotation/annotations/EXPR_AU_Set/Train_Set.csv'
            anno_path = os.path.join(root_dir, anno_relative_path)
            self.label_frame = pd.read_csv(anno_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(anno_path, sep=',', header=0).iloc[:1000]    # for debug
        elif phase == 'validation':
            anno_relative_path = 'Annotation/annotations/EXPR_AU_Set/Validation_Set.csv'
            anno_path = os.path.join(root_dir, anno_relative_path)
            self.label_frame = pd.read_csv(anno_path, sep=',', header=0)
            # self.label_frame = pd.read_csv(anno_path, sep=',', header=0).iloc[:1000]    # for debug
        else: # test
            raise NotImplemented('Test set not implemented yet')

        self.transform = transform

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.label_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        labels_EXPR = self.label_frame.iloc[idx, 1].astype('float')
        labels_EXPR = torch.tensor(labels_EXPR).long()
        labels_AU = self.label_frame.iloc[idx, 2:].astype('float')
        labels_AU = torch.tensor(labels_AU)
        labels = [labels_EXPR, labels_AU]

        return image, labels

    def _gen_matched_label(self, img_dir, label_dir_FER, label_dir_AU):

        video_dir_FER = os.listdir(label_dir_FER)
        video_dir_AU = os.listdir(label_dir_AU)

        video_dir_common = set(video_dir_FER) & set(video_dir_AU)

        label_frame = np.zeros((0, 14))
        for filename in video_dir_common:
            video_folder_name = filename[:-4]
            print(f'processing video {video_folder_name}')
            img_file_name_paths = sorted(glob.glob(os.path.join(img_dir, video_folder_name, '*.jpg')))
            img_label_FER_path = os.path.join(label_dir_FER, filename)
            img_label_AU_path = os.path.join(label_dir_AU, filename)

            file_data = read_process_file('multi-task', img_file_name_paths, [img_label_FER_path, img_label_AU_path])
            label_frame = np.append(label_frame, file_data.label_frame, axis=0)

        label_frame = pd.DataFrame(label_frame,
                                   columns=['path', 'EXPR', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15',
                                            'AU23', 'AU24', 'AU25', 'AU26'])
        label_dir = '/'.join(label_dir_FER.split('/')[:-2])
        set_flag = label_dir_FER.split('/')[-1]
        label_EXPR_AU_dir = os.path.join(label_dir, 'EXPR_AU_Set')
        if not os.path.exists(label_EXPR_AU_dir):
            os.makedirs(label_EXPR_AU_dir)
        label_framw_save_path = os.path.join(label_EXPR_AU_dir, set_flag+'.csv')
        label_frame.to_csv(label_framw_save_path, index=False)
        print(f'label saved to {label_framw_save_path}')


class read_process_file():

    def __init__(self, task, frames, txt_file):

        self.task = task
        self.frames = frames
        self.multi_task_label_frame = []

        if task == 'EXPR':
            self.read_Expr(txt_file)
            self.frames_to_label(self.label_array_EXPR, frames)
        if task == 'AU':
            self.read_AU(txt_file)
            self.frames_to_label(self.label_array_AU, frames)
        elif task == 'multi-task':
            txt_file_EXPR = txt_file[0]
            txt_file_AU = txt_file[1]
            self.read_Expr(txt_file_EXPR)
            self.read_AU(txt_file_AU)
            self.frames_to_label_multi_task()

    def read_Expr(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        lines = lines[1:] # skip first line
        lines = [x.strip() for x in lines]
        lines = [int(x) for x in lines]

        self.label_array_EXPR = np.array(lines)

    def read_AU(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        lines = lines[1:] # skip first line
        lines = [x.strip() for x in lines]
        lines = [x.split(',') for x in lines]
        lines = [[float(y) for y in x ] for x in lines]

        self.label_array_AU = np.array(lines)

    def frames_to_label(self, label_array, frames, discard_value=-1):
        assert len(label_array) >= len(frames)  # some labels need to be discarded
        frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]  # frame_id start from 0
        N = label_array.shape[0]
        label_array = label_array.reshape((N, -1))
        to_drop = (label_array == discard_value).sum(-1)
        drop_ids = [i for i in range(len(to_drop)) if to_drop[i]]
        frames_ids = [i for i in frames_ids if i not in drop_ids]
        indexes = [True if i in frames_ids else False for i in range(len(label_array))]
        label_array = label_array[indexes]
        assert len(label_array) == len(frames_ids)
        # prefix = '/'.join(frames[0].split('/')[:-1])
        prefix = frames[0].split('/')[-2]
        return_frames = [prefix + '/{0:05d}.jpg'.format(id + 1) for id in frames_ids]
        return_frames = np.expand_dims(np.array(return_frames), axis=1)

        self.label_frame = np.concatenate((return_frames, label_array), axis=1)

    def frames_to_label_multi_task(self, discard_value=-1):
        assert len(self.label_array_EXPR) >= len(self.frames) # some labels need to be discarded
        assert len(self.label_array_AU) >= len(self.frames)

        if len(self.label_array_EXPR) > len(self.label_array_AU):
            self.label_array_EXPR = self.label_array_EXPR[0:len(self.label_array_AU)]
        elif len(self.label_array_EXPR) < len(self.label_array_AU):
            self.label_array_AU = self.label_array_AU[0:len(self.label_array_EXPR), :]
        assert self.label_array_EXPR.shape[0] == self.label_array_AU.shape[0]

        frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in self.frames] # frame_id start from 0
        N_FER = self.label_array_EXPR.shape[0]
        label_array_FER = self.label_array_EXPR.reshape((N_FER, -1))
        to_drop_FER = (label_array_FER == discard_value).sum(-1)

        N_AU = self.label_array_AU.shape[0]
        label_array_AU = self.label_array_AU.reshape((N_AU, -1))
        to_drop_AU = (label_array_AU == discard_value).sum(-1)

        to_drop = to_drop_FER | to_drop_AU

        drop_ids = [i for i in range(len(to_drop)) if to_drop[i]]
        frames_ids = [i for i in frames_ids if i not in drop_ids]
        indexes = [True if i in frames_ids else False for i in range(len(label_array_FER))]

        label_array_FER = label_array_FER[indexes]
        label_array_AU = label_array_AU[indexes]
        assert len(label_array_FER) == len(frames_ids)
        assert len(label_array_AU) == len(frames_ids)
        # prefix = '/'.join(self.frames[0].split('/')[:-1])
        prefix = self.frames[0].split('/')[-2]
        return_frames = [prefix+'/{0:05d}.jpg'.format(id+1) for id in frames_ids]
        return_frames = np.expand_dims(np.array(return_frames), axis=1)

        self.label_frame = np.concatenate((return_frames, label_array_FER, label_array_AU), axis=1)


