"""
Preaparing list of files
"""
from glob import glob
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

data_path = './data'
labels = '/labels.csv'

folders = ['/bathroom', '/bedroom', '/dinning_room', '/house', '/kitchen', '/living_room']
val_folder = ['/validation']

def get_data(data_path, folders):
    """
    Params:
    data_path - Path to main folder with data
    folders - list of folders where training data are

    Returns:
    files - list of all training files
    """
    files = []
    for dir in folders:
        for f in glob(data_path + dir + '/*.jpg', recursive=True):
            files.append(f)

    return files


def get_validation_data(data_path, val_folder):
    """
    Params:
    data_path - Path to main folder whith data
    val_folder - list of folders where validation data are

    Returns:
    val_files - list of all validation files
    """
    val_files = [f for f in glob(data_path + '/*.jpg', recursive=True)]

    return val_files


def labels_generation(data_path, labels_path):
    """
    Params:
    data_path - path to data folder in format './folder_name'
    labels_path - name of the labels file, in format '/labels_file_name'

    Returns:
    dict fucking DICCCCCKT
    """
    path = labels_path
    df_labels = pd.read_csv(path)

    classes = []
    df_labels_no_val = df_labels[df_labels.task2_class != 'validation']
    df_labels_val_only = df_labels[df_labels.task2_class == 'validation']

    for idx, row in df_labels_no_val.iterrows():
        if row['task2_class'] not in classes:
            classes.append(row['task2_class'])

    # one hot
    df_labels_no_val['task2_class'] = df_labels_no_val['task2_class'].apply(lambda x: classes.index(x))

    ohe = OneHotEncoder(sparse=False)

    out = {}
    for index, row in df_labels_no_val.iterrows():
        vals = row.values
        rest = vals[1:]
        encoded = np.eye(6)[vals[2]]
        out[vals[0]] = np.concatenate((np.array([vals[1]/4], dtype=float), encoded, np.array([vals[3]/4], dtype=float), np.array(vals[3:])))
    return out


def unparse_labels(data, path):
    """
    params:
    data - numpy array with predicted labels of images size=(56,1)

    Returns:
    data_frame - pandas ...
    """
    classes = ['house', 'dining_room', 'kitchen', 'bathroom', 'living_room', 'bedroom']

    labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                     'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                     'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                     'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                     'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                     'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                     'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                     'Wall', 'Window']
    columns = ["standard", "task2_class", "tech_cond"] + labels_task_1

    predictions = pd.DataFrame(columns=['filename'] + columns)
    p = pd.DataFrame(columns=['filename'] + columns)

    preds = []
    for image in data:
        p = []
        p.append(os.path.basename(image))
        #p['filename'] = image
        #print('len', len(data[image][0]))

        t_class = []
        for i, b in enumerate(data[image][0]):
            if i==0:
                b = int(np.round(b*4))
            elif i>0 and i<8:
                if i==7:
                    b = classes[np.argmax(t_class)]
                else:
                    t_class.append(b)
                    continue
            elif i==8:
                b = int(np.round(b*4))
            elif i>8:
                b = int(np.round(b))
            p.append(b)
        preds.append(p)
    for i, p in enumerate(preds):
        print(p)
        predictions.loc[i] = p

    predictions.to_csv('test.csv', index=False)
