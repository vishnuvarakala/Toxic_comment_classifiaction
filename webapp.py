# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:36:26 2023

@author: vishn
"""
import numpy as np
import pickle
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the saved model
loaded_model = pickle.load(open('D:/Assignments/fall 2023/Jayanth/project/final_train.sav', 'rb'))
load_tfidf = pickle.load(open('D:/Assignments/fall 2023/Jayanth/project/TF_IDF_file.sav', 'rb'))



# Creating a function for prediction
def comment_prediction(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the tokens back into a string
    processed_text = ' '.join(tokens)
    
    # Use the fitted TF-IDF vectorizer
    X_test = load_tfidf.transform([processed_text])  # Note: Pass a list containing the text
    
    prediction = loaded_model.predict(X_test)
    
    if prediction[0] == 0:
        return 'Non Toxic'
    else:
        return 'Toxic Comment'

def main():
    
    # Adding to the application
    st.title('Toxic Comment Classifiaction Web APP')
    
    # getting input from user
    comment = st.text_input('Write a comment')

    
    result = ''
    
    
    if st.button("Comment Results"):
        result = comment_prediction(comment)
        #print(result)
    st.success(result)

if __name__ == '__main__':
    main()
