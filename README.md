## Sitting Posture Recognition

**This is a software that can output the sitting position of a person when the lateral view (side view) of the person is given as input**. 

The output can tell you whether a person is sitting in a *Straight position*, *Hunchback position* (leaning forward), *Reclined position* (leaning backward) and if the person is *Folding Hands* and *Folding Legs* (Kneeling Position).

- This uses the [Open Pose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) model which was invented by [CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/). 
- This OpenPose model can detect the keypoints of the human body. These keypoints co-ordinates can be used to estimate the sitting posture of the person.

![Skeleton detected by OpenPose](https://cdn-images-1.medium.com/max/600/1*oVTetBH6worv5grwvSFkxw.png)

- This software can detect multiple people's keypoints but can only detect the posture of a single person. This can be extended  by simply iterating the detection part of the code over all the set of keypoints that are detected for each person.
- I have used a trained keras model of OpenPose to detect the keypoints. You can download the model from [here](https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5)

- [Here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) is the original implementation by it's authors.
 
- I have tested this software with images of my friend [Venu](https://github.com/vchrombie) sitting in various positions. You can find these images in `sample_images` folder.

## Files
`model.py` - contains architecture of the model.

`config_reader.py` - contains the parameters that are essential for the model to predict the key points. Keeping the specifications of the system in mind.

`util.py` - some functions required to calculate the co-ordinates of the key points.

### Usage 

1. Please install all the requirements from requirements.txt.
2. Run `python3 posture_image.py` for testing this software with an image as input. If you wish to test it with your own image, put that image in `sample_images` folder and change accordingly on `line 237`. 
3. Run `python3 posture_realtime.py` for testing it in real time. Please sit sufficiently far away from the system showing your lateral view. Please note that this will require a system with atleast 8GB RAM. On a 4 GB RAM, the output is not at all smooth and the output lags very much from the input frame.

## Example

For the below image:

![image](https://github.com/nvinayvarma189/Sitting-Posture-Recognition/blob/master/sample_images/img.jpg)

we would get the following output

![output](https://github.com/nvinayvarma189/Sitting-Posture-Recognition/blob/master/output%20images/output.png)


**NOTE**: This curretly works on images and in real-time (through webcam). When used in real life situations, input from webcam (front view of a person) will not work. We can install a camera which captures the lateral view of the person and the output of this camera can be given as the input to real-time version of this software. 

Also, When there are multiple people sitting in a row, the lateral view captures only one person. For this, the camera must be adjusted at a sufficient height so that all the people can be detected. When this is implemented, there shall be slight changes in the angle thresholds through which this software classifies the sitting posture.
