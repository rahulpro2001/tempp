from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
from PIL import Image

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion



# path_to_test_img = 'test_data\pins_Chris Pratt\Chris Pratt14_759.jpg'


def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    try:
        img = Image.open(img_path)
        face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
        emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
        
        saved_data = torch.load('data.pt') # loading data.pt file
        embedding_list = saved_data[0] # getting embedding data
        name_list = saved_data[1] # getting list of names
        dist_list = [] # list of matched distances, minimum distance is used to identify the person
        
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
            
        idx_min = dist_list.index(min(dist_list))
        return (name_list[idx_min], min(dist_list))
    except:
        return ['','']

correct = 0
total = 0

for folder in os.listdir('static/test_data'):
    print(folder)
    for img in os.listdir('static/test_data/'+folder):
        path_to_test_img = 'static/test_data/'+folder+'/'+img
        print(path_to_test_img, end = ' ')
        result = face_match(path_to_test_img, 'data.pt')
        if result[0] == folder:
            print('Matched')
            correct += 1
        else:
            print('Not matched')
        total += 1

print('There are ',correct, ' correct results out of ',total, 'total results')


# print('Face matched with: ',result[0], 'With distance: ',result[1])
# print(result)