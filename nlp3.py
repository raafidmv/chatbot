import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import arabic_reshaper
from bidi.algorithm import get_display

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('arabic'))

# Function to remove even rows
# @st.cache_data
def remove_even_rows(input_file):
    df = pd.read_csv(input_file)
    df = df.iloc[::2]  # Keep only odd rows
    return df

# Function to remove stop words
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to find the most similar question
def find_most_similar_question(input_question, df, column='Tax Enquiry'):
    # Remove stop words from the input question
    cleaned_input_question = remove_stop_words(input_question)

    # Copy the DataFrame to keep the original questions
    df_copy = df.copy()

    # Remove stop words from DataFrame questions
    df_copy[column] = df_copy[column].apply(remove_stop_words)

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the questions
    tfidf_matrix = vectorizer.fit_transform(df_copy[column])

    # Transform the input question
    input_vec = vectorizer.transform([cleaned_input_question])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # Find the index of the most similar question
    most_similar_idx = cosine_sim.argmax()

    # Reshape the Arabic text for proper display
    reshaped_question = arabic_reshaper.reshape(df.iloc[most_similar_idx][column])
    displayed_question = get_display(reshaped_question)

    return df.index[most_similar_idx], displayed_question, cosine_sim[most_similar_idx]

def main():
    # Streamlit app
    st.title("Similar Question Finder (Arabic....)")

    # Fixed file path
    input_file = 'qns_ar.csv'
    df = remove_even_rows(input_file)

    # Get user input
    input_question = st.text_input("Enter your question in Arabic")

    if input_question:
        # Find the most similar question
        most_similar_index, most_similar_question, similarity_score = find_most_similar_question(input_question, df)

        # Display the results
        st.write(f"Input Question: {input_question}")
        st.write(f"Most Similar Question Index: {most_similar_index+2}")
        st.write(f"Most Similar Question: {most_similar_question}")
        st.write(f"Similarity Score: {similarity_score}")

if __name__ == "__main__":
    main()
