from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import json
#changes for json
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
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


ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    details = open("id_to_details.json", "r").read()
    details_into_json = json.loads(details)
    # print(details_into_json)
    current_entries = len(details_into_json)
    id = str(current_entries + 1)
    if request.method == "POST":
        image = request.files["image"]
        details = open("id_to_details.json", "r").read()
        details_into_json = json.loads(details)
        current_entries = len(details_into_json)
        id = str(current_entries + 1)
        name = request.form['name']
        image = request.files['image']
        age = request.form['age']
        phone = request.form['phone']
        description = request.form['description']
        
        
        if not os.path.exists(f"static/train_data/{id}/"):
            os.makedirs(f"static/train_data/{id}/")

        data = {
            "id": id,
            "name": name,
            "photo": f"static/train_data/{id}/{id}_{len(os.listdir('static/train_data/'+id))}.jpg",
            "age": age,
            "where_found": "",
            "description": description,
            "contact": phone
        }

        details_into_json[id] = data
        with open("id_to_details.json", "w") as outfile:
            outfile.write(json.dumps(details_into_json))
        image.save((os.path.join(f"static/train_data/{id}/",f"{id}_{len(os.listdir('static/train_data/'+id))+1}.jpg")))
        
        return redirect("/upload")
    return render_template("upload.html",id=id)

# @app.route("/upload-existing", methods=["GET", "POST"])
# def upload_existing():
#     if request.method == "POST":
#         image = request.files["image"]
#         image.save(secure_filename('uploaded_pic.png'))
#         return redirect("/upload-existing")
#     return render_template("upload_existing.html")

# @app.route("/view-person", methods=["GET", "POST"])
# def view_person():
#     if request.method == "POST":
#         id = request.form["id"]
#         img_names = os.listdir(os.path.join('static/train_data',id))
#         images = []
#         for img in img_names:
#             images.append('/static/train_data/'+id+'/'+img)
#         # retrieve images of the person from server or database
#         # images = get_images(name)
#         # images = id
#         print(id)
#         details = open("id_to_details.json", "r").read()
#         details_into_json = json.loads(details)
#         return render_template("view-person.html", id=id, images=images,name=details_into_json[id]["name"])
#     return render_template("view-person.html")

@app.route("/view-all", methods=["GET", "POST"])
def view_all():
    # retrieve all images from server or database
    
    # images = os.listdir('static/train_data')
    # return render_template("view-all.html", images=images)
    details = open("id_to_details.json", "r").read()
    details_into_json = json.loads(details)
    return render_template("view-all.html", detail = details_into_json)

@app.route("/view-profile/<id>", methods=["GET", "POST"])
def view_profile(id):
    img_names = os.listdir(os.path.join('static/train_data',id))
    images = []
    for img in img_names:
        images.append('/static/train_data/'+id+'/'+img)
    # retrieve images of the person from server or database
    # images = get_images(name)
    # images = id
    print(id)
    details = open("id_to_details.json", "r").read()
    details_into_json = json.loads(details)
    return render_template("view-profile.html", id=id, images=images,name=details_into_json[id]["name"],age=details_into_json[id]["age"],contact=details_into_json[id]["contact"],description=details_into_json[id]["description"])

@app.route("/test-model", methods=["GET", "POST"])
def test_model():
    if request.method == "POST":
        image = request.files["image"]
        # print(image)
        image.save('static/test_image.jpg')
        path_to_test_img = 'static/test_image.jpg'

        

        result = face_match(path_to_test_img, 'data.pt')
        details = open("id_to_details.json", "r").read()
        details_into_json = json.loads(details)
        print('Face matched with: ',result[0], 'With distance: ',result[1])
        print(result)


        
        return render_template('test_model.html',person_matched = details_into_json[result[0]]['name'],first_img=f"static/train_data/{result[0]}/{result[0]}_1.jpg",uploaded_img = "static/test_image.jpg",match_id = result[0])
    return render_template('test_model.html')


@app.route("/train-model")
def train_model():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from PIL import Image

    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

    dataset=datasets.ImageFolder('static/train_data') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    face_list = [] # list of cropped faces from photos folder
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
    total = len(loader)
    # print(loader)
    i=0
    for img, idx in loader:
        if i%10 == 0:
            print(f'{i}/{total} done')
        i+=1
        face, prob = mtcnn(img, return_prob=True) 
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
            name_list.append(idx_to_class[idx]) # names are stored in a list

    print('Saving results....')
    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file
    print('Done....')
    return "Training is done"
        

if __name__ == '__main__':
   app.run(debug = True)