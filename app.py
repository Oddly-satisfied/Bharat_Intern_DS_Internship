import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

cv = CountVectorizer()
tfidf = TfidfVectorizer()
ps = PorterStemmer()
model = MultinomialNB()
encoder = LabelEncoder()

df = pd.read_csv('spam.csv', encoding = 'latin1')

# Drop last 3 columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Renaming the columns
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
df['target'] = encoder.fit_transform(df['target'])

# Removing duplicated values
df = df.drop_duplicates(keep='first')

# No. of characters used by every text
df['num_chars'] = df['text'].apply(len)

# Fetching number of words used in each SMS
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# Fetching Sentences in each SMS
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

# For Spam messages
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

# For Ham messages
ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model.fit(X_train, y_train)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # 1. Preprocessing
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Message is a SPAM")
    else:
        st.header("Message is NOT A SPAM")