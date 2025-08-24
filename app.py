import os, re, string, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Task2 - News Category Classification", layout="wide")
st.title("🗞️ News Category Classification — Advanced")

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return None

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    text_col = None; label_col = None
    for cand in ["text","headline","title","content","article","body"]:
        if cand in cols:
            text_col = df.columns[cols.index(cand)]; break
    for cand in ["category","label","topic","section","class","target"]:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]; break
    return text_col, label_col

def clean_text(s, stopwords):
    if not isinstance(s,str): return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+"," ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    toks = [w for w in s.split() if w.isalpha() and w not in stopwords]
    return " ".join(toks)

def get_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("english")
        except:
            nltk.download("stopwords")
        return set(stopwords.words("english"))
    except:
        return set(["the","a","an","in","on","and","is","it","this","that","of"])

STOPWORDS = get_stopwords()

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (text + category)", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo large CSV", value=True)
vector = st.sidebar.selectbox("Vectorizer", ["TF-IDF","Hashing"], index=0)
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression","Linear SVM","Multinomial NB","Random Forest"], index=0)
ngram = st.sidebar.slider("Max n-gram",1,3,2)
max_feat = st.sidebar.slider("Max features",1000,50000,20000,step=1000)
test_size = st.sidebar.slider("Test size %",10,40,20)/100.0

if uploaded is not None:
    df = load_csv(uploaded)
elif use_demo:
    df = load_csv("data/news_large_sample.csv")
else:
    df = load_csv("data/sample_news.csv")

if df is None or df.empty:
    st.error("No data loaded. Upload CSV or use demo file in data/.")
    st.stop()

text_col, label_col = detect_columns(df)
if text_col is None or label_col is None:
    st.error("Could not detect text/label columns. Columns: " + ", ".join(df.columns))
    st.stop()

st.write("Showing sample rows:")
st.write(df[[text_col, label_col]].sample(min(5,len(df))))

with st.spinner("Cleaning text..."):
    df["_clean"] = df[text_col].apply(lambda x: clean_text(x, STOPWORDS))
    le = LabelEncoder()
    df["_y"] = le.fit_transform(df[label_col].astype(str))
    classes = list(le.classes_)

if vector == "Hashing":
    vec = HashingVectorizer(n_features=max_feat, alternate_sign=False, ngram_range=(1,ngram))
    X = vec.transform(df["_clean"])
else:
    vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1,ngram))
    X = vec.fit_transform(df["_clean"])

y = df["_y"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

with st.spinner("Training model..."):
    if model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000)
    elif model_choice == "Linear SVM":
        clf = LinearSVC()
    elif model_choice == "Multinomial NB":
        clf = MultinomialNB()
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
st.metric("Accuracy", f"{acc*100:.2f}%")
st.metric("Macro F1", f"{f1m*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm)
for (i,j), val in __import__('numpy').ndenumerate(cm):
    ax.text(j,i,int(val),ha='center',va='center')
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
st.pyplot(fig)

if hasattr(clf, "coef_"):
    feats = np.array(vec.get_feature_names_out()) if hasattr(vec, "get_feature_names_out") else None
    if feats is not None:
        st.subheader("Top words per class (weights)")
        import numpy as np
        for idx, cls in enumerate(classes):
            weights = clf.coef_[idx]
            top_ids = np.argsort(weights)[-15:][::-1]
            items = list(zip(feats[top_ids], weights[top_ids]))
            names = [w for w,_ in items]; vals=[float(v) for _,v in items]
            fig, ax = plt.subplots(figsize=(6,3))
            ax.barh(names[::-1], vals[::-1])
            ax.set_title(f"Class: {cls}")
            st.pyplot(fig)

try:
    from wordcloud import WordCloud
    st.subheader("Wordclouds per class")
    for cls in classes:
        text = " ".join(df[df["_y"]==le.transform([cls])[0]]["_clean"].tolist())
        if text.strip():
            wc = WordCloud(width=600, height=300).generate(text)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.imshow(wc); ax.axis('off'); ax.set_title(cls)
            st.pyplot(fig)
except Exception:
    st.info("wordcloud not installed; skipping wordclouds")

st.subheader("Try a headline/text")
txt = st.text_area("Enter text:")
if st.button("Predict"):
    if txt.strip():
        x = vec.transform([clean_text(txt, STOPWORDS)])
        pred = clf.predict(x)[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"Predicted category: {label}")
    else:
        st.warning("Type something to predict")

if st.button("Download test predictions CSV"):
    out = pd.DataFrame({"true": y_test, "pred": y_pred})
    st.download_button("Download CSV", data=out.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime='text/csv')
