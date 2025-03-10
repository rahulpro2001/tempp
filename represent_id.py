'''
    This script was run to change the folder names of folders and files:
    Intially the name was names of people , but we allotted an id to each face 
    Then we stored the folders with id (instead of their names).
    Just running the script does the job
'''
import os
import pprint
import json

curr_dir = 'test_data'
folders = os.listdir(curr_dir)

hash_dict = dict()
curr = 1


for folder in folders:
    path = os.path.join(curr_dir,folder)
    person_name = folder.replace('pins_','')
    # print(person_name)
    hash_dict[str(curr)] = person_name

    old_path = path
    new_path = os.path.join(curr_dir,str(curr))
    os.rename(old_path,new_path)

    curr += 1

'''
# # Don't run these lines again until you are very sure about that

# with open("id_to_name.json", "w") as outfile:
#     outfile.write(json.dumps(hash_dict))
# pprint.pprint(hash_dict)
'''

for folder in os.listdir(curr_dir):
    curr_path = os.path.join(curr_dir,folder)
    c=0
    for files in os.listdir(curr_path):
        # print(folder+'_'+str(c),end=' ')
        c+=1
        old = os.path.join(curr_path,files)
        new = os.path.join(curr_path,folder+'_' + str(c) + '.jpg')
        print(old,new)
        os.rename(old,new)