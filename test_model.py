import os
import cv2 as cv
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = load_model('models/classification/1_model_cats5_e_01_acc_0.09.h5')

dataset_path = os.getcwd() + '\\Dataset\\test'
cat_names = ["ari", "coby", "midas", "pisco", "ruble", "unknown"]
conf = np.arange(0.55, 0.99, 0.05)
unknown_images = 61

for c in conf:
    known_true = 0
    unknown_true = 0
    num_images = 0
    for name in os.listdir(dataset_path):
        cat_path = os.path.join(dataset_path, name)
        files = os.listdir(cat_path)
        #print(name)
        for i in range(len(files)):
            #img = load_img(os.path.join(cat_path, files[i]), target_size=(224, 224))
            frame_x = cv.imread(os.path.join(cat_path, files[i]))
            frame = cv.cvtColor(frame_x, cv.COLOR_BGR2RGB)
            img = cv.resize(frame, (224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            result = model.predict(x)
            cat_array = np.array(result[0])
            num_images += 1
            if (cat_array[cat_array.argmax()] > c) and (cat_names[cat_array.argmax()] == name):
                known_true += 1
            elif (cat_array[cat_array.argmax()] <= c) and (cat_names[len(cat_array)] == name):
                unknown_true += 1

    print("Confidence: " + str(format(c, '.2f'))
          + "\tImages (T=K+U): " + str(num_images) + "=" + str(num_images-unknown_images) + "+" + str(unknown_images)
          + "\tTotal Accuracy: " + str(format((known_true + unknown_true) / num_images, '.4f'))
          + "\tKnown Acc: " + str(format(known_true / (num_images-unknown_images), '.4f'))
          + "\tUnknown Acc: " + str(format(unknown_true/unknown_images, '.4f')))
