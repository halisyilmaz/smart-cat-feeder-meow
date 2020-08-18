# smart-cat-feeder-meow
A smart cat feeder that can identify new cats and plan a special feeding schedule for each cat.

## V2 - Custom Trained Object Detecion Model
Uses only 1 NN. A pretrained SSDLite model is trained with transfer learning to include new cat classes. 
Training for 10 cat classes takes app. 2 hours with an NVIDIA GTX1080 6GB.
Uses TF Object Detection API

## V3 - Custom Trained Image Classifier
Uses 2 NNs. First one is a pretrained SSDLite object detection model to detect cats and dogs in the frame. 
Second one is continuously updated custom trained cat identification network. 
Training for 10 cat classes takes app. 10 minutes with an NVIDIA GTX1080 6GB.
Uses TF2.1/Keras and MobileNetV2 as architecture
