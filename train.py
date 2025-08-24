import os, argparse, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

def clean_text_simple(s):
    import re, string
    if not isinstance(s,str): return ""
    s = s.lower(); s = re.sub(r"http\S+"," ", s); s = re.sub(r"<[^>]+>"," ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join([w for w in s.split() if w.isalpha()])

def main(data, model_dir):
    df = pd.read_csv(data)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "category" if "category" in df.columns else df.columns[1]
    df = df[[text_col,label_col]].dropna().sample(frac=1, random_state=42)
    df["clean"] = df[text_col].apply(clean_text_simple)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(df["clean"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log = LogisticRegression(max_iter=2000); log.fit(X_train, y_train)
    nb = MultinomialNB(); nb.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200); rf.fit(X_train, y_train)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(log, os.path.join(model_dir, "logreg.pkl"))
    joblib.dump(nb, os.path.join(model_dir, "nb.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "rf.pkl"))
    joblib.dump(vec, os.path.join(model_dir, "tfidf.pkl"))
    joblib.dump(le, os.path.join(model_dir, "labelenc.pkl"))
    print("Saved models to", model_dir)
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],)), keras.layers.Dense(256, activation="relu"), keras.layers.Dense(len(np.unique(y_train)), activation="softmax")])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train.toarray(), y_train, epochs=3, batch_size=256, validation_split=0.1)
        model.save(os.path.join(model_dir, "keras_nn"))
        print("Saved Keras model")
    except Exception as e:
        print("TensorFlow not available or failed:", e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/news_large_sample.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    main(args.data, args.model_dir)
