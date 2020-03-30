AUTHOR: AVIGYAN SINHA

This project recognizes the hand gestures of a person in real-time from the video supplied by a webcam and displays the gesture label on the camera frame in real-time, along with the probability of the label being correct. We use a LeNet architecture to classify the 9 hand gestures.

1) lenet.py - defines the LeNet architecture used

2) train_network.py - trains the model on the dataset and saves the trained model to disk

3) test_network.py -  evaluates the trained model on sample gesture images in "images" folder

4) main.py - takes images from  a webcam in real-time, classifies its gesture label, displays the label and probability of it being correct on the input frames in real-time

5) dataset - this folder contains training dataset corresponding to the 9 hand gestures

6) images - this folder contains sample hand gesture images for evaluation

7) mask.model - trained LeNet model saved to disk

USAGE:
python3 main.py

Note: If you are using anything from this repository please consider citing - Malavika Suresh, Avigyan Sinha, Aneesh R P, "Real-Time Hand Gesture Recognition Using Deep Learning", International Journal of Innovations and Implementations in Engineering(ISSN 2454- 3489), 2019, vol 1 