import streamlit as st

import pandas as pd
from collections import Counter
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#from imblearn.under_sampling import NearMiss
import nltk
nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words("portuguese")
nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer() #ESSE STEMMER É FEITO PARA A LINGUA PORTUGUESA
nltk.download("punkt")
from itertools import islice
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
#from goose3 import Goose
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nlp = spacy.load('pt_core_news_sm')


def sentence_tokenizer(sentence):
    return [token.lemma_ for token in nlp(sentence.lower()) if (token.is_alpha & ~token.is_stop)]

def normalizer(sentence):
    tokenized_sentence = sentence_tokenizer(sentence)
    return ' '.join(tokenized_sentence)

#load modelo NB
filename='classifAmericNB.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#load modelo tidf
filename1='tfidfVecAmeric.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))

st.title("Avalie a sua compra")

with st.form(key='includ_avaliacao'):    
    input_avaliacao = st.text_input(label = "Digite sua avaliação sobre o produto ou serviço")    
    input_button_submit = st.form_submit_button("Enviar")

if input_button_submit:
    resultado1=None
    resultado1=loaded_model.predict(loaded_model1.transform([normalizer(input_avaliacao)]))
    if resultado1 is not None:
        st.write(f'Obrigado por contribuir com a sua opinião. A sua avaliação da compra foi classifica automaticamente como: {resultado1}')
    



# st.title("Avalie nossa Empresa")

# with st.form(key='includ_avaliacao'):
#     input_correcao=None
#     input_avaliacao = st.text_input(label = "Digite sua avaliação sobre o produto ou serviço")
#     if input_avaliacao:
#         resultado=nb.predict(tfidf_vecorizer.transform([normalizer(input_avaliacao)]))
#         st.write(f'Classificação automática da avaliação: {resultado}')
#     input_correcao= st.selectbox("Se a classificação da sua avaliação não está correta, por favor reclassifique selecionando uma das opções abaixo", ["Excelente", "Bom", "Regular", "Ruim", "Péssimo"])
#     input_button_submit = st.form_submit_button("Enviar")

# if input_button_submit:
#     st.write(f'Sua Avaliação: {input_avaliacao}')
#     if input_correcao is None:
#         st.write(f'Classificação automática: {resultado}')        
#     else:
#         st.write(f'Classificação manual: {input_correcao}')

