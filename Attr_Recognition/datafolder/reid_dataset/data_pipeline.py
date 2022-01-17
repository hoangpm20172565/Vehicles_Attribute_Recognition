import json
from os.path import join, isdir

# this function will just read the .json file and load in all info
# return 2 dictionaries with (file_name, label) items and labels
def import_attributes(json_file_dir):
    if not isdir(json_file_dir):
        raise ValueError("Label file not existing.")

    file_location = ['train_set.json', 'test_set.json', 'val_set.json']
    attr_set = ['train_attr', 'test_attr', 'val_attr']

    labels = [
         'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'white', 'black',
         'small', 'medium', 'large',
         'bus', 'truck', 'car', 'coach', 'taxi', 'lorry'
    ]

    for attr, file_name in zip(attr_set, file_location):
        globals()[attr] = {}
        with open(join(json_file_dir, file_name), 'r') as file:
            globals()[attr] = json.load(file)

    return train_attr, test_attr, val_attr, labels


# return 2 lists; each list contains a sublist of path to image and id
# data is a list of image path and id - name of the file without the '.jpg'
# ids is just list of id (id in this case is just the file name) -> no need since each id is unique
def import_data(data_dir):
    # subdirectories containing
    classes = ['bus', 'car', 'coach', 'lorry', 'taxi', 'truck']
    groups = ['train_set', 'test_set', 'val_set']
    for group in groups:

        # make a list that contain the sublist [path, name]
        globals()[group] = []

        # load the json file that has all the valid samples from each class
        file_loc = join(data_dir, 'labels', group+'.json')
        valid_data = {}
        with open(file_loc, 'r') as file:
            valid_data = json.load(file)

        subdir = 'train' if group == 'val_set' else group.split('_')[0]
        dpath = join(data_dir, subdir)
        for file in valid_data:
            c = file.split('_')[0]
            globals()[group].append([join(dpath, c, file+'.jpg'), file])
    return train_set, test_set, val_set
