# ICS2000 Group Assigned Practical Task

This application applies filters to the face in real-time based on the emotion detected.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Usage](#usage)

## Repository Structure

- [**Final**](Final/): This directory contains all of the scripts used in the final application, as well as the filter images.
    - [**Filters**](Final/Filters/): Contains all of the filter images.
    - [**emotion_model.keras**](Final/emotion_model.keras): The emotion classifier model.
    - [**frontend.py**](Final/frontend.py): This is the main application.
    - [**filter_engine.py**](Final/filter_engine.py):
    - [**emotionclassifier.py**](Final/emotionclassifier.py): Where the model layers are defined.
    - [**train.py**](Final/train.py): Script used to load the data and train the model.
    - [**model_info.py**](Final/model_info.py): Outputs the model accuracy on the test data.
    - [**requirements.txt**](Final/requirements.txt): Dependency list.

- [**Test**](Test/): This directory contains all of the scripts used while developing and testing the program. This directory contains all of the same files as the ones in [Final](Final/), plus the following two scripts, which do not make part of the final application but were used for testing.
    - [**webcaminput.py**](Test/webcaminput.py): First test on getting input from the webcam, detecting faces and landmarks, and analyzing emotions.
    - [**guitest.py**](Test/guitest.py): Initial test on a GUI application that captures video input and detects faces, landmarks, and emotions.

## Usage

To run the project install all of the dependencies listed in [requirements](Final/requirements.txt).

To run this application you need to download the pretrained landmark detector model from https://drive.google.com/file/d/1ejb8ffOVlg_yURJ--1O8xCYcLOPgAQKn/view?usp=sharing and add it to your project, inside the 'Final' directory or the 'Test' directory. This model could not be uploaded directly on Github due to its large size.

