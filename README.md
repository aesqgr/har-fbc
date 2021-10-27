<p align="center">
<img src="src/dashboard_pages/imgs/logo.png" alt="HAR-Tastic" width="150"/>
</p>

# HAR-TASTIC

HAR-Tastic is Human Activity Recognition (HAR) model that can detect what activity you are performing based on your phone's accelerometers' readings. 

It has been developed as final project for the Machine Learnign and Big Data Bootcamp @ [CORE School](https://www.corecode.school/).

### Access the Project [here]("https://share.streamlit.io/aesqgr/har-fbc/main/dashboard.py")

## About the project

Have you ever wondered how apps like Runtastic or Strava are able to detect what kind of activity you are performing? Whether is running, sleeping or simply walking, your phone or wearable are able to detect what you are doing.

I have implemented a Long Short-term memory, an artificial recurrent neural network architecture for the human activity recognition task.

## About the data 

Your phone is packed with sensors that can be used to monitor your activity. Specifically, they contain tri-axial accelerometers that measure acceleration in all three spatial dimensions. 

The Dataset is obtained from WISDM Lab, Department of Computer & Information Science, Fordham University, NY [here]("https://www.cis.fordham.edu/wisdm/dataset.php"). It was collected from 36 uses as they performed different activities. 


## About the model

From Wikipedia:
``` 
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture \n
used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has \n
feedback connections. It can process not only single data points (such as images), but also \n
entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks \n
such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection \n
in network traffic or IDSs (intrusion detection systems).
```