# SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using text preprocessing, TF-IDF features, and a Linear Support Vector Machine (SVM).

## Features

- Loads the SMS Spam Collection CSV dataset
- Inspects the dataset and checks class distribution
- Cleans SMS text (URLs, email addresses, digits, punctuation, stopwords, and word forms)
- Removes duplicate messages
- Converts text into TF-IDF features
- Trains a Linear SVM classifier
- Tunes the SVM `C` parameter using `GridSearchCV`
- Reports accuracy, precision, recall, F1-score, and cross-validation score
- Displays a confusion matrix
- Visualizes words associated with Spam and Ham
- Lets you test a custom SMS message
- Saves and downloads the trained model as a `.joblib` file

## Dataset

Use the [SMS Spam Collection dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Download the CSV file from Kaggle. The notebook supports the common Kaggle columns `v1` and `v2`, or columns named `label` and `message`.

> The dataset is not included in this repository. Upload the CSV in Google Colab when prompted.

## Run in Google Colab

1. Download `SMS_Spam_SVM_Enhanced.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Choose **File → Upload notebook** and upload the notebook.
4. Run each cell from top to bottom.
5. When prompted, upload the dataset CSV file.
6. Review the metrics and charts, then use the custom SMS cell to test a message.

## Run locally

Recommended: Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m venv .venv
```

Activate the virtual environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook using Jupyter:

```bash
jupyter notebook SMS_Spam_SVM_Enhanced.ipynb
```

Note: The notebook's dataset-upload cell uses `google.colab.files`, so the easiest way to run the complete notebook as written is Google Colab. For local use, replace that upload cell with a local CSV path.

## Project structure

```text
SMS_Spam_SVM_GitHub_Project/
├── SMS_Spam_SVM_Enhanced.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Evaluation

The notebook reports classification metrics on a held-out test set and performs cross-validation during hyperparameter tuning. Results depend on the dataset and environment; run the notebook to generate your actual results.

## Limitations

- A model can make mistakes, especially on unusual, short, or context-dependent messages.
- The decision score is an SVM score, not a calibrated probability.
- Do not use the prediction as the sole basis for blocking important messages.

## Author

Add your name, course/lab section, and instructor details here if required by your course.

## License

Choose a license only if your instructor or project requirements specify one. Otherwise, you can leave this section out.
