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

This project from my coursework in online university skillbox. From there dataset taken, quality of this dataset not so good. Best **accuracy** which i get - around 40%. But i did flexibale code, there many adjustments. If you want to train this model on your own dataset (or on any dataset from net) - you can adjust 10 hyperparameters. Also simple interface done, so - its can be use like simple app. For possibilities with this code pls continue read this readme.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USE PRETRAINED MODEL -->
## Use pretrained model
