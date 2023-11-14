<a name="readme-top"></a>
<h3 align="center">EMOTION_RECOGNITION</h3>
<p align="center">
    neural net for emotion recognition on video
    <br />
    <a href="https://github.com/sanchelo2006/EMOTION_RECOGNITION"><strong>Explore the docs »</strong></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#use-pretrained-model">Use pretrained model</a>
      <ul>
        <li><a href="#requirements-for-enviroment">Requirements for enviroment</a></li>
        <li><a href="#required-files">Required files</a></li>
        <li><a href="#run-file">Run file</a></li>
      </ul>
    </li>
    <li>    
      <a href="#train-model-on-custom-dataset">train model on custom dataset</a>
      <ul>
        <li><a href="#clean-dataset">Clean dataset</a></li>
        <li><a href="#train-model">Train model</a></li>
        <li><a href="#save-result">Save result</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is my approach for emotion recognition on video. This project consist of two neural nets:
- one for **find face** in frame on video (multiface possible, number of faces - argument for run file, see next chapter)
- second for emotion recognition. **ResNet** architecture used.

This project from my coursework in online university skillbox. From there dataset taken, quality of this dataset not so good. Best **accuracy** which i get - around 40%. But i did flexibale code, there many adjustments. If you want to train this model on your own dataset (or on any dataset from net) - you can adjust 10 hyperparameters. Also simple interface done, so - its can be use like simple app. For possibilities with this code pls continue read this readme.\
This model trained for 9 emotion: {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise', 8: 'uncertain'}\

P.S. All comments in code in Russian language. If somebody interest - i can translate and change to English.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USE PRETRAINED MODEL -->
## Use pretrained model

Here is explanation how to run the model on your maschine which i trained already.

### requirements for enviroment

There are 2 files **requirements_pip.txt** and **requirements_conda.txt** with all require libraries for this project.\
If your enviroment installed with pip - use this command: **pip install -r requirements_pip.txt**.\
If your envirement created with conda - use this command: **conda install --file requirements_conda.txt**.

### required files

for run this model you need following files in folder from there you run this model:
- **res10_300x300_ssd_iter_140000.caffemodel** and **deploy.prototxt**. This 2 files need for the model which is responsable for find face in frame on video.
- **zero_detect_face_video.py**. This file for run model. Its have 2 required arguments and 5 with default value.

  required arguments:
  - p (prototxt). Path to the file **deploy.prototxt**
  - m (model). Path to the file **res10_300x300_ssd_iter_140000.caffemodel**\

  not requred arguments:
  - n (numberpixels). Number of pixel for shift. When face moving in frame - its need for correct showing prediction of emotion, otherwise its will be always new face. Default = 60
  - f (frames). Number of frames for averaging prediction. Default = 3
  - c (confidence). Minimum threshold for prediction (predictions with less threshold will not take into account). Default = 0.5
  - v (camera). This is special for openCV library, number of camera for grab video. (0 is for inbuild camera of computer). Default = 0
  - l (maxfaces). Max number of faces in frame. Default = 5

### run file

Run the file **zero_detect_face_video.py** from command line with required arguments (and, if you need other values, other arguments).

Just simple example:
- python zero_detect_face_video.py -p /PATH_TO_YOUR_FOLDER/deploy.prototxt -m /PATH_TO_YOUR_FOLDER/res10_300x300_ssd_iter_140000.caffemodel

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TRAIN MODEL ON CUSTOM DATASET -->
## Train model on custom dataset

Also you can train prebuid model with ResNet architecture on other dataset. The main point is - that you can change hyper parameters.

### clean dataset

This folder also have code for cleaning dataset. All for it inside the file **Preprocessing.py**. This file like a module, you need to import it to your code. Consist of 2 classes: one for augmentation dataset, second for balancing dataset.

Important thing! All your images should be in folders with name of emotion. Function in class **PreprocessDataset** (in module **Preprocessing.py**) has arguments - path tot the folders.\
Folders should be like this:
```
+---MAIN_FOLDER
|  ---ANGER
|  ---CONTEMPT
    ...
|  ---UNCERTAIN
```
Possibilities of module clean:
- crop faces on image (for this used same neural net for find face from openCV zoo)
- balance dataset. If you have much difference in quantity of images of different emotions - this code will balance your dataset by implement augmentation.

### train model
