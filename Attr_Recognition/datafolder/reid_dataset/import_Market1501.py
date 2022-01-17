import os
from .reiddataset_downloader import *

# create 3 global dictionary with the name of the 3 data_group: train, query and gallery (test)
# each dictionary have id of each sample as the keys
# each id have 6 camera angles of it and each angle is stored in different list
# return the 3 dictionaries
def import_Market1501(dataset_dir):
    market1501_dir = os.path.join(dataset_dir,'Market-1501')
    if not os.path.exists(market1501_dir):
        print('Please Download Market1501 Dataset')
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(market1501_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(market1501_dir, 'query')
        else:
            name_dir = os.path.join(market1501_dir, 'bounding_box_test')
        file_list=os.listdir(name_dir)

        # create a global dictionary call group
        globals()[group]={}
        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0] # get the id
                if id not in globals()[group]:
                    # create a list for each id in the group
                    globals()[group][id]=[]
                    # then create 6 empty list as the objects
                    # each of this list is an image that got captured by a different camera angle -> 6 angles
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])            
                    globals()[group][id].append([])

                # get the camera number, minus 1 since the number starts with 0
                cam_n = int(name.split('_')[1][1])-1
                # assign the path of image to the corresponding position in the list
                globals()[group][id][cam_n].append(os.path.join(name_dir,name))
    return train,query,gallery