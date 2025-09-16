import os 
import shutil
import random
import pandas as pd 
from torchvision import transforms
from PIL import Image 

old_folder = "./old_data"
new_folder = './new_data'
new_folder_path = os.path.join(new_folder)

train_folder = "./training_folder"
validation_folder = "./validation"

sample_size = 1000

def delete_image(image_path): 
    if '.txt' in image_path:
        os.remove(image_path)
        return 
       
    img = Image.open(image_path)
    img_shape = transforms.ToTensor()(img).size()
    if img_shape[0] == 1:
        os.remove(image_path)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

for i, folder in enumerate(os.listdir(old_folder)):
    if folder == "test":
        continue 

    subfolders =  os.listdir(os.path.join(old_folder, folder))

    for _subfolder in subfolders:
        subfolder_files = os.listdir(os.path.join(old_folder, folder, _subfolder))
        file_samples = random.sample(subfolder_files, sample_size)

        for _file in file_samples:
            delete_image(os.path.join(old_folder, folder, _subfolder, _file))

        # print(_subfolder)
        # print(file_samples)
        # print("\n")
        
        n1 = int(0.85*sample_size)
        training_images, validation_images = file_samples[:n1], file_samples[n1:] 

        for _path in ["training", "validation"]:
            if not os.path.exists(os.path.join(new_folder, _path)):
                os.makedirs(os.path.join(new_folder, _path))

        for t_img in training_images:
            shutil.copyfile(os.path.join(old_folder, folder, _subfolder, t_img), os.path.join(new_folder, "training", t_img))

        for v_img in validation_images:
            shutil.copyfile(os.path.join(old_folder, folder, _subfolder, v_img), os.path.join(new_folder, "validation", v_img))


new_training_data = os.listdir(os.path.join(new_folder, "training"))
new_validation_data = os.listdir(os.path.join(new_folder, "validation"))

train_val_data = [new_training_data, new_validation_data]
csvs = ["train_dataset.csv", "validation_dataset.csv"]

for data, csv in zip(train_val_data, csvs):
    image_split = len(data)//2
    random.shuffle(data)

    cover_images = data[:image_split]
    secret_images = data[image_split:]

    dataset = []

    for i in range(image_split):
        dataset.append({"cover_image": cover_images[i], "secret_image": secret_images[i]})

    dataframe = pd.DataFrame(dataset)
    dataframe.to_csv(f"./new_data/{csv}")


