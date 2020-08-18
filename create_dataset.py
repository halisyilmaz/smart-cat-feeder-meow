import os
import shutil
import random

dataset_path = os.getcwd() + '\\Dataset'
val_frac = 0.2
test_frac = 0.1


def dataset_gen1():
    try:
        os.mkdir(dataset_path + "\\" + name + "\\train")
        os.mkdir(dataset_path + "\\" + name + "\\val")
        os.mkdir(dataset_path + "\\" + name + "\\test")
    except OSError:
        print("Creation of the directory failed for " + str(name))
    else:
        print("Successfully created the directory for " + str(name))


def dataset_gen2():
    for name in os.listdir(dataset_path):
        try:
            os.mkdir(dataset_path + "\\" + name + "\\train")
            os.mkdir(dataset_path + "\\" + name + "\\val")
            os.mkdir(dataset_path + "\\" + name + "\\test")
        except OSError:
            print("Creation of the directory failed for " + str(name))
        else:
            print("Successfully created the directory for " + str(name))

        cat_path = os.path.join(dataset_path, name)
        if (len(os.listdir(cat_path + "\\train")) == 0) and (len(os.listdir(cat_path + "\\val")) == 0) and (len(os.listdir(cat_path + "\\test")) == 0):
            files = [f for f in os.listdir(cat_path) if f.endswith(".jpg")]
            random.shuffle(files)
            for i in range(len(files)):
                if i < int(len(files)*val_frac):
                    os.rename(os.path.join(cat_path, files[i]),
                              os.path.join(cat_path + "\\val", files[i]))
                elif i < (int(len(files)*val_frac) + int(len(files) * test_frac)):
                    os.rename(os.path.join(cat_path, files[i]),
                              os.path.join(cat_path + "\\test", files[i]))
                else:
                    os.rename(os.path.join(cat_path, files[i]), os.path.join(cat_path + "\\train", files[i]))