import os
from PIL import Image # also this
import torch
from torch.utils import data # read document about this
import numpy as np
from torchvision import transforms as T # and this
# from .reid_dataset import import_MarketDuke_nodistractors       # some dependencies that need to change
# from .reid_dataset import import_Market1501Attribute_binary
# from .reid_dataset import import_DukeMTMCAttribute_binary
from .reid_dataset import import_data, import_attributes


class Train_Dataset(data.Dataset):  # subclassing the data.Dataset class from torch.utils -> Explain

    def __init__(self, data_dir, json_file_dir, transforms=None, train_val='train' ):

        train, gallery, query = import_data(data_dir)    # lists of [path to img, name/id]
        train_attr, test_attr, val_attr, self.label = import_attributes(json_file_dir)

        # if dataset_name == 'Market-1501':
        #     train_attr, test_attr, self.label = import_Market1501Attribute_binary(data_dir) # get the attributes and labels for all the set
        # elif dataset_name == 'DukeMTMC-reID':
        #     train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        # else:   # maybe add in another elif for out self-defined import file
        #     print('Input should only be Market1501 or DukeMTMC')

        self.num_ids = len(train)    # number of ids
        self.num_labels = len(self.label)   # number of labels

        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels)    # Explain to get the distribution of labels in the training set:
        for k, v in train_attr.items(): # need to check out train_attr configuration -> done
            distribution += np.array(v)
        self.distribution = distribution / len(train_attr)


        # get the correct data set based on train_val input
        if train_val == 'train':
            self.train_data = train # get the data info
            # self.train_ids = train['ids']   # get the image id
            self.train_attr = train_attr
        elif train_val == 'query':  # Explain? -> validation set
            self.train_data = query
            # self.train_ids = query['ids']
            self.train_attr = val_attr
        elif train_val == 'gallery':    # Explain? -> Test set
            self.train_data = gallery
            # self.train_ids = gallery['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or val')

        self.num_ids = len(self.train_data) # get the number of id


        # perform some basic transformation on the input data (resize->random flip(for training)->make tensor->normalize)
        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),  # Will resize ruin the image?
                    T.RandomHorizontalFlip(), # randomly flipping the training data
                    T.ToTensor(),   # Explain
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # Explain
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):   # override getitem
        '''
        一次返回一张图片的数据
        '''
        # all of the information of a sample is stored in a list under seld.train_data
        img_path = self.train_data[index][0]
        # i = self.train_data[index][1]   # id of the image
        id = self.train_data[index][1]  # id of the person
        # cam = self.train_data[index][3]  # the camera id

        # get label from id
        label = np.asarray(self.train_attr[id])

        # open image and proccess
        data = Image.open(img_path)
        data = self.transforms(data)

        # get the name of the sample
        # name = self.train_data[index][4]
        return data, label, id

    def __len__(self):  # get the number of data of the current set
        return len(self.train_data)

    def num_label(self): # get the number of labels
        return self.num_labels

    def num_id(self):   # number of ids
        return self.num_ids

    def labels(self):   # get the label names
        return self.label



class Test_Dataset(data.Dataset):
    def __init__(self, data_dir, json_file_dir, transforms=None, query_gallery='gallery' ):
        train, gallery, query = import_data(data_dir)  # lists of [path to img, name/id]
        train_attr, test_attr, val_attr, self.label = import_attributes(json_file_dir)

        # if dataset_name == 'Market-1501':
        #     self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        # elif dataset_name == 'DukeMTMC-reID':
        #     self.train_attr, self.test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        # else:
        #     print('Input should only be Market1501 or DukeMTMC')

        if query_gallery == 'query':
            self.test_data = query
            # self.test_ids = query['ids']
            self.test_attr = train_attr
        elif query_gallery == 'gallery':
            self.test_data = gallery
            # self.test_ids = gallery['ids']
            self.test_attr = test_attr
        elif query_gallery == 'all':
            self.test_data = gallery['data'] + query['data']
            # self.test_ids = gallery['ids']
            self.test_attr = val_attr
        else:
            print('Input shoud only be query or gallery;')

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data[index][0]
        id = self.test_data[index][1]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        # name = self.test_data[index][4]
        return data, label, id

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label
