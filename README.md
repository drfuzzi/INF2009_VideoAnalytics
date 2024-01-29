**Video Analytics with Raspberry Pi 4 using Web Camera**

**Objective:** By the end of this session, participants will understand how to set up a web camera with the Raspberry Pi 4, capture video streams, and perform basic video analytics.

---

**Prerequisites:**
1. Raspberry Pi 4 with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. Web camera compatible with Raspberry Pi.
4. Internet connectivity (Wi-Fi or Ethernet).
5. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
- Video analytics is an emerging field employed to extract valuable insights from video data. Edge video analytics with real-time processing capabilities is chellenging but important and inevitable due to privacy/security concerns. Also, in many cases redundancy can be avoided to save on the bandwidth requirements (e.g. compress the video to have only key (important) frames). In this lab, few basic and advanced video processing tasks on edge devices is introduced. An overview of the experiments/setup is as follows:
![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/882c84dc-1989-4039-807d-554a079e3776)

**2. Setting up the Raspberry Pi (10 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```

**3. Connecting and Testing the Web Camera (5 minutes)**
- Please ensure the web camera is working and proceed to subsequent steps.

**4. Introduction to real-time video processing on raspberry pi (20 minutes)**
- **Ensure you will be running the subsequent steps in a new virtual environment.**
-  [MediaPipe](https://developers.google.com/mediapipe) is a framework for building cross platform multimodal applied ML pipelines that consist of fast ML inference, classic computer vision, and media processing (e.g. video decoding). MediaPipe was open sourced at CVPR in June 2019 as v0.5.0.
- Installing OpenCV and media pipe:
  ```bash
  pip install opencv-python
  pip install mediapipe
  ```
- Hand landmark detection
  - Download the handlandmark detection model:
    ```bash
    wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    ``` 
  - The [sample code](Codes/hand_landmark.py) employs opencv and mediapipe to detect the human hand and subsequently the finger locations (the tip of thumb and index finger as well as a simple logic to predict if the thumb is pointing up) based on the [finger model](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) outlined below :
    ![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/1090e213-7a56-4059-9386-50123bd6f8f8)
  - Modify the code to show all the 21 finger points and observe the same while moving the hand.
  - Modify the code to predict the number of fingers and display the same overlaid on the image as text (e.g. if four fingers are raised, display '4' on the screen and if three fingers on one hand and two on the other, the display should be '5').


**5. Basic Video Analytics (40 minutes)**
- Motion detection: Identifying movement in the video stream.
- Face detection in real-time using OpenCV's pre-trained classifiers.
- Basic object tracking based on color or features.
- Extracting frames from the video stream and processing them.

**6. Advanced Video Analytics (20 minutes)**
- Introduction to more advanced techniques:
  - Background subtraction for detecting moving objects.
  - Utilizing pre-trained deep learning models for object recognition in videos.

---

**Homework/Extended Activities:**
1. Build a surveillance system that sends alerts based on motion detection.
2. Integrate video analytics with IoT devices, e.g., turning on lights when motion is detected.
3. Experiment with more advanced tracking algorithms available in OpenCV.

---

**Resources:**
1. Raspberry Pi official documentation.
2. OpenCV documentation and tutorials.
3. Online communities and forums related to Raspberry Pi and video analytics.

---
