import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from googletrans import Translator

# Download NLTK stopwords
nltk.download('stopwords')
stop_words_english = set(stopwords.words('english'))

# Initialize Translator
translator = Translator()

# Example DataFrame with columns 'questions'
data = {
    'questions': [
        'What is income tax?',
        'What is a bank guarantee?',
        'What are the details about a bank guarantee?',
        'Why is there a notification for claims and obligations when I opened the system to get the tax card?',
        'What should a taxpayer do if there is a mistake in submitting the return?',
        'Why does the old name of the company appear on the tax card after changing the name with the Ministry of Commerce and Industry?',
    ]
}
df = pd.DataFrame(data)

# Function to correct the input question
def correct_question(input_question):
    blob = TextBlob(input_question)
    corrected_question = str(blob.correct())
    return corrected_question

# Function to remove stop words
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words_english]
    return ' '.join(filtered_words)

# Function to find the most similar question
def find_most_similar_question(input_question, df, column='questions'):
    # Translate input question to English
    translated_input = translate_text(input_question)
    
    # Translate Arabic questions to English
    df['translated_questions'] = df['questions'].apply(lambda x: translate_text(x) if not all(ord(char) < 128 for char in x) else x)
    
    # Correct the input question
    corrected_question = correct_question(translated_input)
    cleaned_corrected_question = remove_stop_words(corrected_question)
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the questions
    tfidf_matrix = vectorizer.fit_transform(df['translated_questions'])
    
    # Transform the input question
    input_vec = vectorizer.transform([cleaned_corrected_question])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Find the index of the most similar question
    most_similar_idx = cosine_sim.argmax()
    
    return df.iloc[most_similar_idx], cosine_sim[most_similar_idx]

# Function to translate text
def translate_text(text, src_lang='ar', dest_lang='en'):
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

# Streamlit app
def main():
    st.title("Question Similarity Finder")

    input_question = st.text_input("Enter your question in Arabic:", "ماهي ضريبة الدخل")

    if st.button("Find Similar Question"):
        most_similar_question, similarity_score = find_most_similar_question(input_question, df)
        st.write("Input Question:", input_question)
        # st.write("Translated Input Question:", translate_text(input_question))
        # st.write("Corrected Input Question:", correct_question(translate_text(input_question)))
        st.write("Most Similar Question:", most_similar_question['questions'])
        st.write("Similarity Score:", similarity_score)

if __name__ == "__main__":
    main()
