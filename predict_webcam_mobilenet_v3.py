import os
import cv2 as cv
import time
import numpy as np
from collections import deque
from scipy import stats
from datetime import datetime
from random import uniform

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

cap = cv.VideoCapture("cat.mp4")  # Change to "test_video.mp4" for video / 0 for webcam / 1 for external camera
similarity_threshold = 0.7

def print_fps(start, frame):
    end = time.time()
    if (end - start) > 0.016:
        fps = round(1 / (end - start), 2)
    else:
        fps = 60
    cv.putText(frame, str(fps), (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)


def plot_model(model):   # Print model architecure
    plot_model(model, to_file='model.png')
    print(model.summary())


def update_logs(cat_name, weight, cup):
    with open(r"C:\Users\halis\PycharmProjects\Meow Website\catlist\management\commands\input.txt", "a") as log_file:
        log_file.write("\n")
        log_file.write(str(cat_name) + "; " + str(weight) + " kg; " + str(cup) + " cup; "
                       + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(r"C:\\Users\halis\PycharmProjects\Meow Website\catlist\management\commands\input.txt", "r") as log_file:
        log_data = log_file.readlines()
        status = log_data[0].split("; ")
        new_food = int(status[1].split('\\')[0])
        new_charge = int(status[0])
        if new_food > 5 and new_charge > 5:
            new_food -= cup
            new_charge -= 1
        log_data[0] = str(new_food) + "; " + str(new_charge) + "\n"
    with open(r"C:\\Users\halis\PycharmProjects\Meow Website\catlist\management\commands\input.txt", "w") as log_file:
        log_file.writelines(log_data)


def feed_cat(cat_name, cup, frame):
    cv.putText(frame, "FEED " + str(cat_name) + str(cup) + " CUP", (30, 120),
               cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


def get_latest_model():
    files = os.listdir("models/classification")
    for file in files:
        if file.endswith(".h5"):
            model_name = (os.path.join(r"models\classification", file))
    return model_name

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cvNet = cv.dnn.readNetFromTensorflow('models/detection/ssdlite_frozen_graph.pb', 'models/detection/ssdlite_graph.pbtxt')
model = load_model(str(get_latest_model()))


classes = [] 
with open("models/detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
with open("models/classification/cat_names.txt", "r") as f:
    cat_names = [line.strip() for line in f.readlines()]
cat_names.append("unknown")
print(cat_names)
layer_names = cvNet.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in cvNet.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cat_queue = deque(maxlen=20)    # Which cat is most probably
dog_queue = deque(maxlen=20)    # Dog exist of not

follow_cat = None
tracker_isInit = False
isClassified = False
save_count = 0
lastFedTime = datetime.now()
showFeed = 20


while True:
    start = time.time()
    ret, frame = cap.read()
    # Object Detection Network
    height, width, channels = frame.shape

    if follow_cat is not None:
        # grab the new bounding box coordinates of the object
        tracker_isInit = True
        (success, box) = tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]

        # check to see if the tracking was a success
        if success:
            if isClassified: # If 10 of the cat classifier results out of 20, cat is classified
                # Log the new cat
                weight = uniform(1.0, 3.5)
                with open(r"C:\\Users\halis\PycharmProjects\Meow Website\catlist\management\commands\input.txt", "r")\
                        as log_file:
                    lines = log_file.readlines()
                firstTime = True
                for line in reversed(lines):
                    cat_data = line.split("; ")
                    if cat_data[0] == cat_names[cat_mode[0][0]]:
                        #print(str(cat_names[cat_mode[0][0]]) + " is last fed at " + str(cat_data[3]))
                        lastFedTime = datetime.strptime(cat_data[3][:-1], '%d/%m/%Y %H:%M:%S')
                        firstTime = False
                        break
                fedAgo = datetime.now() - lastFedTime
                print("Cat: "+str(cat_names[cat_mode[0][0]])+" Last visit was " +str(fedAgo.total_seconds()) + "sec ago")
                if firstTime or fedAgo.total_seconds() > 600:  #In seconds
                    feed_cat(cat_names[cat_mode[0][0]], round(weight, 1), frame) # Send feeding signal to arduino here
                    showFeed = 0
                    update_logs(cat_names[cat_mode[0][0]], round(weight, 1), int(weight))
                if showFeed < 20:    # For demonstration purposes
                    cv.putText(frame, "FEED " + str(cat_names[cat_mode[0][0]]), (30, 120),
                               cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    showFeed += 1

                if save_count % 5 == 0:
                    cv.imwrite(filename='Saved Cats/cat_' + str(cat_mode[0][0]) + '_%d.jpg' % save_count, img=frame)
                    cv.putText(frame, "SAVE " + str(cat_names[cat_mode[0][0]]) + " IMAGE", (30, 150),
                               cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    #print("[INFO] Image saved.")

                cv.putText(frame, "Cat: " + str(cat_names[cat_mode[0][0]]), ((int(x) + 75), int(y)), cv.FONT_HERSHEY_PLAIN,
                           2, colors[class_id], 2)

                save_count = save_count + 1

                # Print bounding box
                cv.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], thickness=2)

                print_fps(start, frame)
                cv.imshow("Frame", frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                # Image Classification Network
                frame_x = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = cv.resize(frame_x, (224, 224))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                result = model.predict(x)
                cat_array = np.array(result[0])
                #print("Classification Result: " + str(cat_array) + " Cat:" + str(cat_names[cat_array.argmax()])
                      #+ " / Prob: " + str(cat_array[cat_array.argmax()]))
                if cat_array[cat_array.argmax()] > similarity_threshold: # Unknwon Threshold
                    cat_queue.append(cat_array.argmax())
                    #print("Car added " + str(cat_names[cat_array.argmax()]) + str(cat_array[cat_array.argmax()]))
                else:
                    #print("Unknown is cat " + str(len(cat_array)))
                    cat_queue.append(len(cat_array))
                cat_mode = stats.mode(cat_queue)

                # If 10 of the cat classifier results out of 20, cat is classified
                if cat_mode[1][0] > 10:
                    #print("10/" + str(cat_mode[1][0]) + "\tIt is " + str(cat_names[cat_mode[0][0]]))
                    isClassified = True

                print_fps(start, frame)
                cv.imshow("Frame", frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        else:
            follow_cat = None
            image_count = 0
            cat_queue.clear()
            isClassified = False
            fed = False

    cvNet.setInput(cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        score = detection[2]
        class_id = int(detection[1]) - 1

        if score > 0.8:
            left = detection[3] * width
            top = detection[4] * height
            right = detection[5] * width
            bottom = detection[6] * height

            if class_id == 16:      # Cat Detected
                cv.putText(frame, str(round(score, 2)), (int(left), int(top)), cv.FONT_HERSHEY_PLAIN, 2, colors[class_id],
                           2)
                cv.putText(frame, "Cat Detected", ((int(left) + 75), int(top)), cv.FONT_HERSHEY_PLAIN,
                           2, colors[class_id], 2)
                cv.rectangle(frame, (int(left), int(top)), (int(right),int(bottom)), colors[class_id], thickness=2)

                follow_cat = (int(left), int(top), int(right-left), int(bottom-top))
                #print(follow_cat)
                tracker = cv.TrackerKCF_create()
                tracker.init(frame, follow_cat)

            elif class_id ==17:     # Dog detected.
                dog_queue.append(1)
                dog_mode = stats.mode(dog_queue)

                # If 10 of the cat classifier results out of 20, cat is classified
                if dog_mode[1][0] > 8:
                    #print("10/" + str(dog_mode[1][0]) + "\tDog Alarm!")
                    cv.putText(frame, "DOG ALARM", (30, 90), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), colors[class_id], thickness=2)
                    cv.putText(frame, "Dog Detected", ((int(left) + 75), int(top)), cv.FONT_HERSHEY_PLAIN,
                               2, colors[class_id], 2)
    dog_queue.append(0)

    print_fps(start, frame)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()