# Football Match Analysis Using YOLO 📊⚽


[![GitHub issues](https://img.shields.io/github/issues/iiakshat/football_yolo.svg)](https://github.com/iiakshat/football_yolo/issues)
[![GitHub forks](https://img.shields.io/github/forks/iiakshat/football_yolo.svg)](https://github.com/iiakshat/football_yolo/network)
[![GitHub stars](https://img.shields.io/github/stars/iiakshat/football_yolo.svg)](https://github.com/iiakshat/football_yolo/stargazers)
[![GitHub license](https://img.shields.io/github/license/iiakshat/football_yolo.svg)](https://github.com/iiakshat/football_yolo/blob/main/LICENSE)

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Input & Output](#input--output)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [Contributions](#contributions)

## Description
The __Football Match Analysis Using YOLO__ project is an AI-powered system designed to analyze football match videos, extracting player positions, speeds, and distances covered. By leveraging YOLO for object detection and roboflow [football dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) for fine-tuning, this project assigns team colors, tracks ball possession, and adjusts camera movements to provide a comprehensive view of player performance and match dynamics.


## Features
- __Player Tracking and Team Assignment:__ Detects players and assigns them to teams based on jersey color.
- __Speed and Distance Calculation:__ Estimates player speeds and distances covered using advanced tracking algorithms.
- __Ball Possession Detection:__ Identifies which player has possession of the ball at each frame.
- __Camera Movement Adjustment:__ Adapts player positioning for dynamic camera angles using a perspective transformer.
- __Flask Interface:__ Provides an interface for users to upload match clips and customize tracking options for the final output.


## Requirements
- **Python**: 3.11.5
- **Dependencies**: Install all required dependencies from the `requirements.txt` file.

## Project Structure
```
│
├── assignment/              # Team and ball assignment functions
├── matches/                 # Folder for input match videos
├── matrix/                  # Contains accuracy, precision, etc matrices
├── models/                  # Pretrained models
├── movement_estimator/      # Camera movement and player speed estimation modules
├── output/                  # Folder for output videos
├── research/                # Research and analysis resources
├── runs/                    # Contains run data for tracking experiments
├── trackers/                # Player and ball tracking functionality
├── training/                # Model training resources and configurations
├── transformer/             # View transformation for accurate positioning
├── utils/                   # Utility functions for video processing andsaving
├── static/                  # Contains static data (Css, Scripts)
├── templates/               # Contains template to render
├── main.py                  # Execute `main.py` to get output without GUI
├── app.py                   # Flask Application
├── config.json              # Contains predefined configurations
├── .
```

## Input & Output
On Launching `app.py`, you'll see this GUI interface,

![{C13E7DEA-7D5C-4D7F-A326-2B1B635A2A79}](https://github.com/user-attachments/assets/66d2ce01-215f-49a7-9a02-b6e6e5a37c61)


### Customization
Check `Advanced` feature to customize your output.

![{E1F8C0C9-E00D-4A29-BF31-52C56BB9E7E9}](https://github.com/user-attachments/assets/14b0e59a-a48b-4d33-9883-59908205e77c)

- **Input**: Raw video of a football match.

[![{536D3249-5756-4A8A-BA9D-0F0F03CE9D00}](https://github.com/user-attachments/assets/f754b2ad-f51a-4d5c-8969-cf5c40826479)](https://github.com/user-attachments/assets/f52fd141-5b34-4d57-b007-99935dda8530)

- **Output**: Annotated video with player statistics overlayed for each frame.

[![{301B6C62-8F9A-48F1-9BD7-EE664E694046}](https://github.com/user-attachments/assets/e68b9f3f-af3d-4165-a140-1e37cd2f6a75)](https://github.com/user-attachments/assets/e7a2b080-8b28-44a9-a4aa-17304cada8c6)

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Football-Match-Analysis-Using-YOLO
   ```

2. **Set up a virtual environment (recommended)**:
```bash
python3 -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
4. **Update Configurations**:
- Open `config.json` and update parameters like model paths, video titles, and other constants used in the project.

## Usage
1. Run `main.py` or `app.py`
```bash
python main.py
```
2. Check Output folder.

## Model Performance & Evaluation
YOLO model was fine-tuned on a football dataset to classify objects more accurately.

**Dataset**: [Football Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
- __Confusion Matrix__
  ![confusion_matrix](https://github.com/user-attachments/assets/01c7907e-db6b-4f5f-acdc-fe738bb3c815)

- __F1 & Confidence and Precision & Recall Curve__

![F1_curve](https://github.com/user-attachments/assets/dbea5803-5267-47ac-857a-1082c444805e)
![PR_curve](https://github.com/user-attachments/assets/435b17d0-4414-429b-b672-d767b5f30cbb)

**Note**: Due to lack of samples, PR and F1 Score for ball and background seems to very less than expected.


## Contributions

I welcome contributions to enhance this project! Feel free to fork the repository, make improvements, and create a pull request. Suggestions for new features or optimizations are also encouraged.

Thank you for using this project.
