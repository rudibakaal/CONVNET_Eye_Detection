# CONVNET_Eye_Detection

## Motivation
Convolutional neural network designed to detect the difference between images of open vs closed eyes. This detection is important in the field of autonomous driving for instance, to ensure drivers aren't asleep behind the wheel. 

The data set contains 4852 images of eyes falling into 2 classes open and shut, selected from the Labeled Face in the Wild (LFW) database.[1].

## Neural Network Topology and Results Summary

The binary crossentropy loss function was leveraged along with the Rmsprop optimizer for this classification task.

![model_plot](https://user-images.githubusercontent.com/48378196/111556181-09182980-87de-11eb-8425-8c1b593cb1d0.png)

After 10 epochs, the training and validation set classifiers reach ~ 93% accuracy in distinguishing images of open vs closed eyes. 

![eye_data_img](https://user-images.githubusercontent.com/48378196/111556310-5b594a80-87de-11eb-922d-36d04ed2a3ae.png)

## License
MIT 

## References
[1]  Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.
http://vis-www.cs.umass.edu/lfw/#reference 
