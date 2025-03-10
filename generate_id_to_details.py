import json
import os


hash_dict = dict()
with open("id_to_name.json", "r") as infile:
    ff = infile.read()
    hash_dict = json.loads(ff)

# Person = {
#     id,
#     name,
#     photo,
#     age,
#     where found,
#     description,
#     contact(if any),

# }

res_dict = dict()
for key in hash_dict:
    res_dict[key] = {
        "id": key,
        "name": hash_dict[key],
        "photo": f"static/train_data/{key}/" + os.listdir(f'static/train_data/{key}')[0],
        "age": "",
        "where_found": "",
        "description": "",
        "contact": ""
    }

with open("id_to_details.json", "w") as outfile:
    outfile.write(json.dumps(res_dict))
print(res_dict)