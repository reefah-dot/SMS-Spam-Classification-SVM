# SMS Spam Classification Using SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using text preprocessing, TF-IDF feature extraction, and a Linear Support Vector Machine (SVM).

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies and Libraries](#technologies-and-libraries)
- [Workflow](#workflow)
- [Features](#features)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [How to Run in Google Colab](#how-to-run-in-google-colab)
- [How to Run Locally](#how-to-run-locally)
- [Example Prediction](#example-prediction)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

SMS spam messages may contain unwanted advertisements, fraudulent offers, or suspicious links. The goal of this project is to build a text-classification model that identifies whether an SMS is spam or a legitimate message (ham).

The notebook loads and explores the dataset, preprocesses the messages, converts text into numerical features with TF-IDF, and trains a Linear SVM classifier. It also tunes the model's `C` parameter using GridSearchCV and evaluates performance on a held-out test set.

## Objectives

- Prepare SMS text for machine learning.
- Convert text messages into numerical features using TF-IDF.
- Train a Linear Support Vector Machine classifier.
- Tune the SVM hyperparameter `C` using cross-validation.
- Evaluate performance using standard classification metrics.
- Visualize the confusion matrix and words associated with spam and ham.
- Allow users to test a custom SMS message.
- Save the trained model for later use.

## Dataset

This project uses the **SMS Spam Collection Dataset**, available on Kaggle:

[SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains SMS messages labeled as:
- **Ham** — a legitimate, non-spam message.
- **Spam** — an unwanted or spam message.

The notebook supports the common Kaggle column names `v1` and `v2`, as well as `label` and `message`. The dataset file is not included in this repository; download it from Kaggle and upload the CSV file when prompted.

## Technologies and Libraries

- **Python** — programming language
- **Pandas and NumPy** — data handling
- **NLTK** — stopword removal and lemmatization
- **Regular expressions (`re`) and `string`** — text cleaning
- **Scikit-learn** — TF-IDF, Linear SVM, GridSearchCV, train/test split, and evaluation metrics
- **Matplotlib and Seaborn** — visualizations
- **WordCloud** — word visualization
- **Joblib** — saving the trained model
- **Jupyter Notebook / Google Colab** — development environment

## Workflow

1. **Load the dataset:** Read the CSV file and select the message and label columns.
2. **Inspect the data:** Check the dataset structure, missing values, label distribution, and duplicate messages.
3. **Preprocess the text:** Convert messages to lowercase; remove URLs, email addresses, digits, and punctuation; remove English stopwords; and apply lemmatization.
4. **Remove duplicates:** Drop duplicate dataset rows.
5. **Encode labels:** Map `ham` to `0` and `spam` to `1`.
6. **Split the data:** Divide the data into training and testing sets using an 80/20 split and stratification.
7. **Extract features:** Use `TfidfVectorizer` with a maximum of 5,000 features and unigrams and bigrams (`ngram_range=(1, 2)`).
8. **Train and tune:** Train a `LinearSVC` model and use `GridSearchCV` with five-fold cross-validation to compare `C` values of `0.1`, `1`, and `10`, selecting by F1-score.
9. **Evaluate:** Report metrics on the held-out test set.
10. **Explore and use the model:** Display visualizations, predict a custom SMS, and save the trained pipeline.

## Features

- Dataset loading and exploratory data analysis
- Text cleaning and duplicate removal
- TF-IDF feature extraction
- Linear SVM classification
- Hyperparameter tuning with GridSearchCV
- Accuracy, precision, recall, F1-score, and cross-validation results
- Confusion matrix visualization
- Visualization of words associated with spam and ham
- Custom SMS prediction
- Save and download the trained model as a `.joblib` file

## Model Evaluation

The notebook evaluates the model using:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Of the messages predicted as spam, how many were actually spam.
- **Recall:** Of the actual spam messages, how many the model detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions for each class.
- **Cross-validation F1-score:** Helps compare hyperparameter settings across multiple folds of the training data.

Run the notebook to see the actual scores produced by your execution. Results may vary with dataset preparation, library versions, and model settings. Do not report a score in a submission unless it matches the output from your final run.

## Project Structure

```text
SMS-Spam-Classification-SVM/
├── SMS_Spam_SVM_Enhanced.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The CSV dataset is downloaded separately and uploaded when running the notebook.

## How to Run in Google Colab

Google Colab is recommended because the notebook includes a Google Colab file-upload step.

1. Download or open `SMS_Spam_SVM_Enhanced.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload notebook** and upload the notebook, or open it from GitHub if available.
4. Run the notebook cells from top to bottom.
5. When prompted, upload the dataset CSV downloaded from Kaggle.
6. Review the evaluation metrics and visualizations.
7. Use the custom SMS prediction section to test a message.
8. Run the model-saving cell if you want to download the trained `.joblib` file.

## How to Run Locally

Use Python 3.10 or newer.

### 1. Clone the repository

```bash
git clone https://github.com/reefah-dot/SMS-Spam-Classification-SVM.git
cd SMS-Spam-Classification-SVM
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook SMS_Spam_SVM_Enhanced.ipynb
```

**Note:** The notebook's upload cell uses `google.colab.files`, which is designed for Google Colab. To run locally, replace that cell with code that reads the downloaded CSV from its local path.

## Example Prediction

The notebook includes a function for entering a custom SMS message and returning a **Spam** or **Ham (Not Spam)** prediction. Run the relevant training and prediction cells before using it.

Example message:
```text
Congratulations! You have won a free prize. Call now!
```

The output depends on the trained model and should be checked in the notebook.

## Limitations

- The model can misclassify unusual, short, ambiguous, or context-dependent messages.
- Removing digits, punctuation, URLs, or stopwords may discard information that could be useful for some messages.
- The SVM decision score is **not a calibrated probability**.
- This project is educational and should not be used as the sole basis for blocking important messages or making security decisions.

## Future Improvements

- Compare Linear SVM with Multinomial Naive Bayes and Logistic Regression.
- Explore additional preprocessing and TF-IDF settings.
- Evaluate precision and recall carefully, especially for spam detection.
- Test the model on new messages from a separate source.
- Add a simple web interface for interactive predictions.

## Author

**Name:** Reefah  
**Project:** SMS Spam Classification Using SVM  
**GitHub:** [reefah-dot](https://github.com/reefah-dot)

Add your course name, department, or instructor details here if your submission requires them.

## License

This repository does not specify a license. Add one only if you intend to license the project and your course requirements permit it.
