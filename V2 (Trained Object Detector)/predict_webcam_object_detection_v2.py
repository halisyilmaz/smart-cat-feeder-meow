import os
import cv2 as cv
import time
import numpy as np
from collections import deque
from scipy import stats


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cvNet = cv.dnn.readNetFromTensorflow('models/detection/ssdlite_cat8_frozen_graph.pb', 'models/detection/ssdlite_cat8_graph.pbtxt')


with open("models/detection/cat8.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cat_queue = deque(maxlen=10)    # Which cat is most probably
dog_queue = deque(maxlen=10)    # Dog exist of not

tracker = cv.TrackerMOSSE_create()
follow_cat = None
tracker_isInit = False
save_count = 0

cap = cv.VideoCapture(0)  # Change to "test.mp4" for video / 0 for webcam / 1 for external camera
time.sleep(1)

while (1):
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
            # If 10 of the cat classifier results out of 20, cat is classified
            if save_count % 10 == 0:
                #cv.imwrite(filename='Saved Cats/cat_' + str(classes[class_id]) + '_%d.jpg' % save_count, img=frame)
                print("[INFO] Image saved.")

            cv.putText(frame, "Cat " + str(classes[class_id]), ((int(x) + 80), int(y)), cv.FONT_HERSHEY_PLAIN,
                       2, colors[class_id], 2)

            save_count = save_count + 1

            # Print bounding box
            cv.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], thickness=2)

            end = time.time()
            cv.putText(frame, str(round(1 / (end - start), 2)), (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            cv.imshow("Frame", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            follow_cat = None
            image_count = 0
            cat_queue.clear()


    cvNet.setInput(cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    print("[INFO] Object Detection is running")

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        class_id = int(detection[1])-1
        if class_id != 0:
            if score > 0.9:
                print(class_id)
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height
                cv.putText(frame, str(round(score, 2)), (int(left), int(top)), cv.FONT_HERSHEY_PLAIN, 2, colors[class_id], 2)
                cv.putText(frame, str(classes[class_id]), (int(left)+80, int(top)), cv.FONT_HERSHEY_PLAIN, 2, colors[class_id], 2)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), colors[class_id], thickness=2)

                #follow_cat = (int(left), int(top), int(right - left), int(bottom - top))
                #tracker = cv.TrackerKCF_create()
                #tracker.init(frame, follow_cat)
        else:
            if score > 0.4:
                print(class_id)
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height
                cv.putText(frame, str(round(score, 2)), (int(left), int(top)), cv.FONT_HERSHEY_PLAIN, 2, colors[class_id], 2)
                cv.putText(frame, str(classes[class_id]), (int(left)+80, int(top)), cv.FONT_HERSHEY_PLAIN, 2, colors[class_id], 2)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), colors[class_id], thickness=2)


    end = time.time()
    cv.putText(frame, str(round(1 / (end - start), 2)), (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()