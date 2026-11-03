
# SMS Spam Classifier

An NLP and Machine Learning project that classifies SMS messages as Spam or Ham.

## Features

- SMS text classification
- Text preprocessing using stemming and stopword removal
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Interactive Streamlit web application

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- TF-IDF
- Multinomial Naive Bayes

## Project Structure

```text
SMS_Spam_classifier/
├── app.py
├── train_model.py
├── text_preprocessing.py
├── data/
├── artifacts/
├── requirements.txt
├── README.md
└── dockerfile
```

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the application:

```bash
streamlit run app.py
```

## Model

The project uses TF-IDF for text feature extraction and Multinomial Naive Bayes for SMS classification.

## Dataset

SMS Spam Collection dataset.
