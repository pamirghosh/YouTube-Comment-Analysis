from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from langdetect import detect
import pandas as pd
import emoji
import spacy
import joblib
import re

def convert_emojis_to_text(text):
    return emoji.demojize(text, delimiters=(":", ":"))

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return '2'
    elif score['compound'] <= -0.05:
        return '0'
    else:
        return '1'

def preprocess(text, nlp):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'#', '', text)        
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    text = re.sub(r'â|€|œ', '', text)    
    text = text.strip()
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)
    
if __name__=="__main__":
    df=pd.read_csv("D:/YouTube Sentimetn Analysis/sentiment_analysis/dataset_with_vectors.csv")
    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load('en_core_web_sm')
   
    df['comment_text_with_emoji_text'] = df['comment_text'].apply(convert_emojis_to_text)
    df['sentiment'] = df['comment_text_with_emoji_text'].apply(lambda x: get_vader_sentiment(x, analyzer))
    df['cleaned_text'] = df['comment_text_with_emoji_text'].apply(lambda x: preprocess(x, nlp))

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    X = tfidf_matrix
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print('Accuracy:', accuracy_score(y_train, y_pred))

    joblib.dump(model, 'YouTube_Sentiment_Analysis.pkl')