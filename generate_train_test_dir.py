'''
Run this script to create the folders of test and train datasets storing

'''

import shutil 
import os 
path = 'splited_105_face_recognition/test'
test_folder_path = 'test_data'
train_folder_path = 'train_data'

if not os.path.exists('test_data'):
    os.mkdir('test_data')

if not os.path.exists('train_data'):
    os.mkdir('train_data')

# COUNT_IN_TRAIN = 100
# COUNT_IN_TEST = 20

for folder in os.listdir(path):
    c = 0   
    
    COUNT_IN_TRAIN = 0
    COUNT_IN_TEST = len(os.listdir(os.path.join(path,folder)))
    print(folder)
    # if not os.path.exists(os.path.join('train_data',folder)):
    #     os.mkdir(os.path.join('train_data',folder))
    if not os.path.exists(os.path.join('test_data',folder)):
        os.mkdir(os.path.join('test_data',folder))

    for file in os.listdir(os.path.join(path,folder)):
        # ...
        if c < COUNT_IN_TRAIN:
            dest_folder = os.path.join((os.path.join(train_folder_path,folder)),file)
            
        elif c < COUNT_IN_TRAIN + COUNT_IN_TEST:
            dest_folder = os.path.join((os.path.join(test_folder_path,folder)),file)
        
        else:
            break

        src_dir = os.path.join((os.path.join(path,folder)),file)
        dst_dir = dest_folder
        shutil.copy(src_dir,dst_dir)
        c += 1