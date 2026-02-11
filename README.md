## Instagram-Review-Sentiment-Analysis-Using-NLP


<img width="447" height="272" alt="image" src="https://github.com/user-attachments/assets/cff0c514-939a-4295-ad76-3a0dda51a6b9" />


# Project Overview

This project performs Natural Language Processing (NLP) and Machine Learning classification on Instagram Play Store reviews to analyze user sentiment and predict ratings (1–5 stars) based on textual feedback.


This repository demonstrates:

* End-to-end NLP pipeline
* Text preprocessing & cleaning
* TF-IDF feature engineering
* Multi-class classification
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Model saving for deployment


# Problem Statement

Can we predict a user’s rating (1–5 stars) based only on their written review?<br>

This is a multi-class text classification problem where:<br>
Input: Review text<br>
Output: Rating (1, 2, 3, 4, 5)<br>


# Dataset Information

* Source: Kaggle – Instagram Play Store Reviews
* Format: CSV
* Target Variable: rating
* Feature Used: review_description


| Category         | Tools Used                    |
| ---------------- | ----------------------------- |
| Programming      | Python 3                      |
| Data Processing  | Pandas, NumPy                 |
| NLP              | NLTK                          |
| Visualization    | Matplotlib, Plotly, WordCloud |
| Machine Learning | Scikit-learn                  |
| Model Saving     | Pickle                        |


## Exploratory Data Analysis (EDA)

# Rating Distribution

* Visualized distribution of 1–5 star ratings
* Identified class imbalance trends
* Majority reviews are 4 and 5 stars

# Word Analysis

Generated WordCloud for:

* 5-star reviews

<img width="1347" height="676" alt="image" src="https://github.com/user-attachments/assets/e2a02ed1-ccbc-419b-b810-41cc90f4be64" />

* 1-star reviews

<img width="1362" height="693" alt="image" src="https://github.com/user-attachments/assets/63d7b79e-19a3-4277-980b-c94af59154cc" />

* Extracted Top 10 most frequent words


<img width="1043" height="670" alt="image" src="https://github.com/user-attachments/assets/96f16584-4829-4f9d-b955-f5de12e4265d" />

# Key Observations

* Positive reviews contain words like: love, amazing, great
* Negative reviews contain words like: bug, problem, issue




## Text Preprocessing Pipeline

The following steps were applied:<br>

* Lowercasing
* URL removal
* Mention removal (@username)
* Accent normalization
* Punctuation removal
* Stopword removal (NLTK)
* Tokenization
* Stemming (Snowball Stemmer)

Output → Cleaned corpus ready for vectorization.

<img width="517" height="277" alt="image" src="https://github.com/user-attachments/assets/9a8c7ef2-b04e-4f3f-937f-b57ff0cc6321" />



## Feature Engineering

TF-IDF Vectorization
TF-IDF (Term Frequency – Inverse Document Frequency) was used instead of raw count vectors.<br>

Why TF-IDF?<br>

Reduces impact of common words<br>
Gives higher importance to meaningful words<br>
Improves model performance<br>


## Machine Learning Models

| Model                   | Type          |
| ----------------------- | ------------- |
| Multinomial Naive Bayes | Probabilistic |
| Logistic Regression     | Linear        |
| Linear SVM              | Margin-based  |



## Model Evaluation

Evaluation Metrics Used<br>

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross Validation
