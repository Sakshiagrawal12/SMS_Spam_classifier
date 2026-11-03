
import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from text_preprocessing import transform_text


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/spam.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1"
)

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())


# ==========================================
# 2. RECONSTRUCT SMS MESSAGES
# ==========================================

# Original dataset contains:
# v1 = label
# v2 = main SMS message
# Unnamed columns = additional message text
# in some rows due to CSV formatting.

message_columns = [
    "v2",
    "Unnamed: 2",
    "Unnamed: 3",
    "Unnamed: 4"
]

df["message"] = (
    df[message_columns]
    .fillna("")
    .astype(str)
    .apply(
        lambda row: " ".join(
            value.strip()
            for value in row
            if value.strip()
        ),
        axis=1
    )
)

df["label"] = (
    df["v1"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[["label", "message"]]


# ==========================================
# 3. DATA CLEANING
# ==========================================

df.dropna(inplace=True)

df["message"] = df["message"].str.strip()

df = df[
    df["label"].isin(["ham", "spam"])
]

df = df[
    df["message"] != ""
]

# Remove duplicate messages.
df.drop_duplicates(
    subset=["label", "message"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)

print("\nCleaned dataset shape:", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())


# ==========================================
# 4. ENCODE TARGET LABELS
# ==========================================

# Ham = 0
# Spam = 1

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 5. TEXT PREPROCESSING
# ==========================================

df["transformed_text"] = (
    df["message"].apply(transform_text)
)

# Remove messages that become empty
# after preprocessing.

df = df[
    df["transformed_text"].str.strip() != ""
]

print("\nPreprocessing completed.")


# ==========================================
# 6. FEATURES AND TARGET
# ==========================================

X = df["transformed_text"]
y = df["label"]


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(
    max_features=3000
)

X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)

print("\nTF-IDF vectorization completed.")
print(
    "Feature matrix shape:",
    X_train_tfidf.shape
)


# ==========================================
# 9. MODEL TRAINING
# ==========================================

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

print("\nModel training completed.")


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. SAVE MODEL AND VECTORIZER
# ==========================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

artifact = {
    "model": model,
    "vectorizer": tfidf,
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
}

joblib.dump(
    artifact,
    "artifacts/spam_classifier.pkl"
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("Location: artifacts/spam_classifier.pkl")
print("===================================")
