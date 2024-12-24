from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from scipy.sparse import hstack
import pickle
import streamlit as st
import os
import newspaper


def extract_from_url(url):
    article = newspaper.Article(url)
    try:
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return "Title not found", "Content not found"

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def load_vectorizers():
    txtvec_path = './models/vectorizers/txtvec.pkl'
    titlevec_path = './models/vectorizers/titlevec.pkl'

    if not os.path.exists(txtvec_path):
        raise FileNotFoundError(f"Vectorizer file not found: {txtvec_path}")
    if not os.path.exists(titlevec_path):
        raise FileNotFoundError(f"Vectorizer file not found: {titlevec_path}")

    with open(txtvec_path, 'rb') as txtvectorizer_file:
        text_vectorizer = pickle.load(txtvectorizer_file)
    with open(titlevec_path, 'rb') as titlevectorizer_file:
        title_vectorizer = pickle.load(titlevectorizer_file)
    
    return text_vectorizer, title_vectorizer


def vectorize_text(texts, titles, text_vectorizer, title_vectorizer):
    vectorized_texts = text_vectorizer.transform(texts)
    vectorized_titles = title_vectorizer.transform(titles)
    return hstack([vectorized_texts, vectorized_titles])

def predict_fake_news(url, model, text_vectorizer, title_vectorizer):
    title, text = extract_from_url(url)
    
    if not title or not text or title == "Title not found" or text == "Content not found":  
        return 'Invalid URL or content not found.'
    
    preprocessed_text = preprocess_text(text)
    preprocessed_title = preprocess_text(title)
    
    vectorized_text = vectorize_text([preprocessed_text], [preprocessed_title], text_vectorizer, title_vectorizer)
    
    try:
        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)
        
        if prediction[0] == 1:
            return f'Fake (confidence: {probability[0][1]:.2f})'
        elif prediction[0] == 0:
            return f'Real (confidence: {probability[0][0]:.2f})'
    except Exception as e:
        return f"Prediction error: {str(e)}"



def load_model(model_name):
    model_paths = {
        'gradient boosting': './models/gb.pkl',
        'random forest': './models/rf.pkl',
        'adaboost': './models/ab.pkl'
    }

    model_path = model_paths.get(model_name)
    if model_path and os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)
    else:
        raise FileNotFoundError(f"Model file not found for {model_name}")
def main():
    st.title('Fake News Detection')
    url = st.text_input('Enter the URL of the news article:')
    model_name = st.selectbox('Select a model:', ['Random Forest', 'Gradient Boosting', 'AdaBoost'])

    if not url:
        st.write("Please enter a valid URL.")
    else:
        try:
            model = load_model(model_name.lower())
            text_vectorizer, title_vectorizer = load_vectorizers()
            
            if st.button('Predict'):
                prediction = predict_fake_news(url, model, text_vectorizer, title_vectorizer)
                st.write(f'The news article is: {prediction}')
        except FileNotFoundError as e:
            st.write(str(e))
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
