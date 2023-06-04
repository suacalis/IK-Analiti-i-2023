import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import pandas as pd

st.title("Anahtar Kelime Eşleme ve Benzerlik Analizi")

job_posting_file = st.file_uploader("İş İlanınızı Yükleyin")
course_description_file = st.file_uploader("Ders İçeriğini Yükleyin")

if job_posting_file is not None and course_description_file is not None:
    job_posting = job_posting_file.read().decode('utf-8')
    course_description = course_description_file.read().decode('utf-8')

    job_posting_tokens = word_tokenize(job_posting.lower())
    course_description_tokens = word_tokenize(course_description.lower())

    # Anahtar Kelime Eşleme
    vectorizer = TfidfVectorizer().fit_transform([job_posting, course_description])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]

    # Benzerlik Analizi
    job_posting_words = set(job_posting_tokens)
    course_description_words = set(course_description_tokens)
    intersection = job_posting_words.intersection(course_description_words)
    similarity = len(intersection) / len(job_posting_words)

    # Sonuçları DataFrame olarak sakla
    data = {"Metrik": ["Anahtar Kelime Eşleme", "Benzerlik Analizi"],
            "Skor": [cosine_sim, similarity]}
    df_results = pd.DataFrame(data)

    # Grafikle sonuçları göster
    st.bar_chart(df_results.set_index("Metrik"))
