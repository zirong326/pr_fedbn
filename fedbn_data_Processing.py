"""
This file is used to pre-process all data in defact dataset.
i.e., splitted data into train&test set  in a stratified way.
The function to process data into 10 partitions is also provided.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import pickle as pkl
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import  Counter
from tqdm import tqdm
import zipfile


def stratified_split(X, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Train:', Counter(y_train))
        print('Test:', Counter(y_test))

    return (X_train, y_train), (X_test, y_test)                




def process_data(name):
    file_paths = {
        'CM1': r"C:\Users\27746\Desktop\40张紫容\NASADefectData\MDP\CM1.arff",
        'Dataset2': r"path\to\your\Dataset2.file",
        'Dataset3': r"path\to\your\Dataset3.file",
        'Dataset4': r"path\to\your\Dataset4.file",
        'Dataset5': r"path\to\your\Dataset5.file",
        'Dataset6': r"path\to\your\Dataset6.file"
    }

    if name not in file_paths:
        raise ValueError("Invalid dataset name provided.")

    with open(file_paths[name]) as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        data = pd.read_csv(f, header=None)
        data.columns = header

    data['Defective'] = data['Defective'].replace({'N': False, 'Y': True})

    all_label = data['Defective']
    all_feature = data.drop(labels=['Defective'], axis=1)

    train_stratified, test_stratified = stratified_split(all_feature, all_label)
    print('# After spliting:')
    print('Train data:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test data:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)
    
    
    # Ensure directory exists or create it
    data_dir = r'C:\Users\27746\Desktop\40张紫容\data\{}'.format(name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save train and test data
    with open(os.path.join(data_dir, 'train.pkl'), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(data_dir, 'test.pkl'), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def split(data_path, num_partitions=6):
    try:
        # Load the dataset
        with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
            datas, labels = pkl.load(f)
        
        # Print shape of loaded data (for debugging)
        print(f"Loaded datas shape: {datas.shape}, labels shape: {labels.shape}")
        
        # Calculate the size of each partition
        part_len = len(datas) // num_partitions
        
        datas = np.array(datas)  # Convert to numpy array if not already
        labels = np.array(labels)  # Convert to numpy array if not already

        print(f"datas type: {type(datas)}, datas shape: {datas.shape}")
        print(f"labels type: {type(labels)}, labels shape: {labels.shape}")
   
        # Split the dataset into partitions
        for num in range(num_partitions):
            print(f"part_len: {part_len}, num: {num}")
            print(f"Indices: {int(part_len*num)} to {int(part_len*(num+1))}")
            datas_part = datas[int(part_len*num):int(part_len*(num+1)), :]
            labels_part = labels[int(part_len*num):int(part_len*(num+1))]
            
            print(f"Dataset has been split into {num_partitions} partitions.")
            
            # Create a directory to save partitions if it doesn't exist
            save_path = os.path.join(data_path, 'partitions')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Save each partition
            with open(os.path.join(save_path, f'train_part{num}.pkl'), 'wb') as f:
                pkl.dump((datas_part, labels_part), f, pkl.HIGHEST_PROTOCOL)
                
            print(f"Partition {num+1} saved successfully.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__  == '__main__':
    
    print('Processing...')        
     process_data()
    

    
    base_paths = [
        'C:\Users\27746\Desktop\40张紫容\data\CM1'
    ]
    for path in base_paths:
        print(f'Spliting {os.path.basename(path)}')
        split(path)
