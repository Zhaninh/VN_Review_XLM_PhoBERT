<h1 align="center">VN_Review_XLM_PhoBERT 💬</h1>


## Table of contents
- [Overview](#overview)
    - [Project](#project)
    - [Benefits](#benefits)
    - [Workflow](#workflow)
    - [Input and Output](#input-and-output)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Usage](#usage)
- [FastAPI LocalHost Applicaiton](#fastapi-localhost-application)


## Overview

**Project:**

- The primary aim of this project is to create a machine learning model for text classification and regression. It serves as a powerful tool to streamline data analysis and make informed decisions based on text content.

**Benefits:**

- This project offers significant advantages to both individuals and businesses. It saves time by automating text data analysis, empowers decision-making, and provides personalized insights. For businesses, it extracts valuable customer feedback, enhancing efficiency, and offering a competitive edge. 

**Workflow:**

- The project follows a structured workflow, including data preprocessing, model selection, training, evaluation, saving, and testing. Users can leverage the saved model for predictions based on their input text data.

**Input and output for training, and testing:**
- Input: file .csv ***in your language*** with structure:
    - Review, entertainment, accommodation, restaurants, dining, transportation, shopping
- Output: F1 Score, R2 Score, and Final Score (view [metrics.py](./A/deploy/metrics.py))


**Input and Output of localhost application:**

- Input: Reviews from users. Could be a paragraph or a sentence ***in any languages***.
- Output: Aspect categorization and ratings derived from the Reviews.

This project harnesses the potential of textual data, translating it into actionable insights that benefit both individuals and businesses.


## Installation
```bash
git clone https://github.com/Zhaninh/Review_analysis_DreamTim.git

cd ./A/deploy

pip install -r requirements.txt
```

## Preprocessing
**Text Cleaning:**
- Remove special characters from the text.
- Remove punctuation from the text.
- Remove emojis from the text.
- Remove URLs and file paths from the text.
- Normalize annotations, converting "ks" to "khách sạn" (hotel) and "nv" to "nhân viên" (staff).
- Remove escape characters such as line breaks, tabs, and carriage returns.
- Combine the above cleaning methods to create a cleaned review.
  
**Segmentation and Tokenization:**
- Segment the text using VnCoreNLP.
- Tokenize the segmented text using a specified tokenizer.
- Label the data for regression and classification tasks.
- Remove Vietnamese stopwords from the text.

## Usage
- Open file app.py
- Run in terminal:
```bash
uvicorn app:app --host=0.0.0.0 --port=8000 --workers=1
```
- Waiting for application startup.
- Open web browser --> Search:
```bash
https://localhost8000.com
```

## Project Run Guideline
- [Split Train-Dev](./A/deploy/nam_split.py)
- Create a folder named 'Review_analysis_training' in your drive
- [Train and Evaluate](https://colab.research.google.com/drive/1v7PelQhAJtzPIDl2V9qhCaLe6UO9f5Q9?usp=sharing)
  (Adjust 'switch' to select model in file 'training.py')(Ensure that 'switch' variable in 'evaluate.py' file matches the one in 'training.py')
- Download weight into local
- [Testing on application](./A/deploy/app.py)
  ***(Pasting the path leads to the weight - you've just downloaded - into 'MODEL_PATH')***

## Weights
- Weight of XLM in [here](https://drive.google.com/file/d/15yhqZeTRkAXsnuZkB1yxH6rH6iUNgAuV/view?usp=sharing)
- Weight of PhoBERT in [here](https://drive.google.com/file/d/17CHUy43r29bc8azu9p-zT85CAu-V5QKa/view?usp=sharing)

## FastAPI LocalHost Application 
![](./images/Web.PNG)
