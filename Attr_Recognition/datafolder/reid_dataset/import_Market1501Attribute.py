import os
from .import_Market1501 import *
from .reiddataset_downloader import *
import scipy.io


def import_Market1501Attribute(dataset_dir):
    dataset_name = 'Market-1501/attribute'
    train,query,test = import_Market1501(dataset_dir) # 3 dictionaries with id as keys and list of path of each angle as items
    if not os.path.exists(os.path.join(dataset_dir,dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label=['age',
           'backpack',
           'bag',
           'handbag',
           'downblack',
           'downblue',
           'downbrown',
           'downgray',
           'downgreen',
           'downpink',
           'downpurple',
           'downwhite',
           'downyellow',
           'upblack',
           'upblue',
           'upgreen',
           'upgray',
           'uppurple',
           'upred',
           'upwhite',
           'upyellow',
           'clothes',
           'down',
           'up',
           'hair',
           'hat',
           'gender']
    
    test_label=['age',
           'backpack',
           'bag',
           'handbag',
           'clothes',
           'down',
           'up',
           'hair',
           'hat',
           'gender',
           'upblack',
           'upwhite',
           'upred',
           'uppurple',
           'upyellow',
           'upgray',
           'upblue',
           'upgreen',
           'downblack',
           'downwhite',
           'downpink',
           'downpurple',
           'downyellow',
           'downgray',
           'downblue',
           'downgreen',
           'downbrown'
           ]

    # create of sorted list of id (each id contain 6 more list of path to the different angles of that id)
    train_person_id = []
    for personid in train: # when we loop like this, we only get the key of the dictionary
        train_person_id.append(personid)
    train_person_id.sort(key=int) # a list retain all of the ids in training set

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    test_person_id.remove('-1')
    test_person_id.remove('0000')

    # load the attribute stored in .mat file
    # in our case the attributes are stored in .json format
    # more about f: f is an numpy structured array with size of the matlab struct and has 2-D shape
    # for more information, visit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
    f = scipy.io.loadmat(os.path.join(dataset_dir,dataset_name,'market_attribute.mat'))

    test_attribute = {}
    train_attribute = {}

    # this whole for loop is for 
    for test_train in range(len(f['market_attribute'][0][0])):  # get the number of the first element in the market_attribute struct
        if test_train == 0:
            id_list_name = 'test_person_id' # define like this for easy access to test_person_id through locals() function
            group_name = 'test_attribute'   # this is another dictionary with id as key and list of attribute as items
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'

        # loop through each attribute; all attributes are in 2D array 1 x number_of_ids
        for attribute_id in range(len(f['market_attribute'][0][0][test_train][0][0])):  # attribute added to the group_name dict follow and order
            if isinstance(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
                continue
            for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][attribute_id][0])): # the id is also added in order
                id = locals()[id_list_name][person_id] # get the id from dictionary id_list_name
                if id not in locals()[group_name]: # if not appeared yet then make a list for that id
                    locals()[group_name][id]=[]
                # so the group name also have id as key and list of attribute as items
                locals()[group_name][id].append(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])

    # getting the train attribute dictionary to synchronize with the test label
    unified_train_atr = {}
    for k,v in train_attribute.items(): # k is for id, v is a list containing the label
        temp_atr = [0]*len(test_label)
        for i in range(len(test_label)):
            temp_atr[i]=v[train_label.index(test_label[i])]
        unified_train_atr[k] = temp_atr
    
    return unified_train_atr, test_attribute, test_label


def import_Market1501Attribute_binary(dataset_dir):
    train_market_attr, test_market_attr, label = import_Market1501Attribute(dataset_dir)


    #  maybe this code is for expanding the age attribute since it may receive many possible values (4 to be exact)
    for id in train_market_attr:
        train_market_attr[id][:] = [x - 1 for x in train_market_attr[id]] # have to reassign the label since matlab use (1,2)
        if train_market_attr[id][0] == 0:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 1)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 1:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 1)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 2:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 1)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 3:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 1)

    for id in test_market_attr:
        test_market_attr[id][:] = [x - 1 for x in test_market_attr[id]]
        if test_market_attr[id][0] == 0:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 1)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 1:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 1)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 2:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 1)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 3:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 1)


    # yep here are some expanding labels for attribute age -> this make the labels all have binary value
    label.pop(0)
    label.insert(0,'young')
    label.insert(1,'teenager')
    label.insert(2,'adult')
    label.insert(3,'old')
    
    return train_market_attr, test_market_attr, label