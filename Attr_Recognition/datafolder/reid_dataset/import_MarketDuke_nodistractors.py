import os
from .reiddataset_downloader import *


def import_MarketDuke_nodistractors(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir,dataset_name)   # make the path to the data se
    
    if not os.path.exists(dataset_dir):
        print('Please Download '+dataset_name+ ' Dataset')
        
    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','gallery']    # 3 group of data

    # prepare path for ecah group
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir, 'query')
        else:
            name_dir = os.path.join(dataset_dir, 'bounding_box_test')
        file_list=sorted(os.listdir(name_dir)) # sort all the files in the group

        # define global a dictionary for each group with keys: data, ids, data and lists as items
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
            if name[-3:]=='jpg':    # all the information can be retrieved from image name
                id = name.split('_')[0]
                cam = int(name.split('_')[1][1])    # Explain: the camera angle of the images with id (there are 6 angles)
                images = os.path.join(name_dir,name)    # create a path to the image
                if (id!='0000' and id !='-1'): # invalid id
                    if id not in globals()[group]['ids']:   # if the current id is not in group['ids'] already
                        globals()[group]['ids'].append(id)  # add current id into ids

                    # each object in data list is a list containing:
                    #       - image path
                    #       - index of id in ids (image's id)
                    #       - id (human's id)
                    #       - cam
                    #       - name of the file (without .jpg part)
                    globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, gallery # return 3 groups