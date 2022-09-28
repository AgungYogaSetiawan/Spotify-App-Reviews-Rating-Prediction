# Import librari yang diperlukan
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json
import nltk # from nltkmodules.py 
import re
import spacy
import string
from pandas import option_context
from st_aggrid import AgGrid
# import en_core_web_sm
from keras.models import load_model
from langdetect import detect
from googletrans import Translator 
from google_play_scraper import Sort, reviews, app
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from wordcloud import WordCloud, STOPWORDS
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')
# page layout menjadi wide
st.set_page_config(page_title='Spotify Reviews Scraping & Prediction Rating Web App',layout='wide', page_icon=":speech_balloon:")

# Membuat function model building
@st.cache(allow_output_mutation=True)
def model_build(data):
  model = load_model('./saved_model/model_spotify_bert.h5', custom_objects={'KerasLayer':hub.KerasLayer})
  df1, series = data
  y_pred = model.predict(df1['Reviews']) # nantinya nama kolomnya Reviews
  y_pred = [np.argmax(i) for i in y_pred]
  df_pred = df1.copy()
  df_pred['Rating'] = y_pred
  df = pd.concat([series,df_pred['Rating']], axis=1)
  df['index'] = np.arange(1, df.shape[0] + 1)
  df = df[['index', 'Reviews', 'Rating']]
  df.rename(columns={'index':'No'}, inplace=True)
  return df

# membuat fungsi membersihkan data text
def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string
  

# membuat function deteksi bahasa, translate, dan scraping data review dari google play store
def translate_scrape(target_lang='en'):
  # scrape review di play store dengan google play scraper 
  id_app = 'com.spotify.music'
  app_reviews = []
  
  for score in list(range(1,6)):
    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
      rvs, _ = reviews(
        id_app, 
        lang='id', 
        country='id', 
        sort=sort_order, 
        count = 200 if score == 3 else 100,
        filter_score_with=score
      )
      
      for r in rvs:
        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
        r['appId'] = id_app
      app_reviews.extend(rvs)
  
  # membuat dataframe dari hasil scraping kode diatas
  app_reviews_df = pd.DataFrame(app_reviews)
  # st.dataframe(app_reviews_df.head(20)) # menampilkan 20 sampel dataset

  # applying function clean_string to content data
  app_reviews_df['Reviews'] = app_reviews_df['content'].apply(lambda x: clean_string(x, stem='Stem'))
  app_reviews_shuffle_df = app_reviews_df.sample(frac=1, random_state=1).reset_index()
  
  # deteksi bahasa dan translate bahasa dari dataframe hasil scraping
  text_reviews = []
  reviews_df = []
  translator = Translator()
  text_reviews.append(app_reviews_df.Reviews.values.tolist())

  for i in range(250):
    translate_text = translator.translate(text_reviews[0][i],dest='en')
    reviews_df.append(translate_text.text)
  
  reviews_df = pd.DataFrame(reviews_df, columns=['Reviews'])
  
  return reviews_df, app_reviews_shuffle_df['Reviews'][:250]


# Main Menu Web
st.write("""
## :speech_balloon: Spotify Reviews Scraping & Prediction Rating Web App

Web App ini bertujuan untuk memprediksi rating dari ulasan hasil dari scraping google play store pada halaman ulasan Aplikasi Spotify.
""")

# # Menu sidebar 
# with st.sidebar.header('Download Dataset Hasil Scraping'):
#   csv = translate_scrape().to_csv(index=False).encode('utf-8')
#   st.sidebar.download_button(
#     label='Download Dataset Scraping',
#     data=csv,
#     file_name='reviews_df.csv',
#     mime='text/csv',  
#   )
  
# with st.sidebar.header('Setting Paramater'):
#   param_range = st.sidebar.number_input('Masukan jumlah data scrape',1,284)
  

if st.button('Scrape!'):
  data = translate_scrape()
  data = model_build(data)
  df_download = data.to_csv().encode('utf-8')
  # fungsi download dataframe
  st.download_button(
  label="Download data as CSV",
  data=df_download,
  file_name='df_reviews.csv',
  mime='text/csv',
  )
  # menampilkan tabel dataframe
  st.header('Dataset hasil scraping')
  AgGrid(data, height=500, fit_columns_on_grid_load=True)
  
  # menampilkan bar plot
  fig_bar = plt.figure(figsize=(10,10))
  ax = plt.axes()
  ax.bar(data=data, x=data.Rating.value_counts().index, height=data.Rating.value_counts())
  ax.set_xlabel('Rating')
  ax.set_ylabel('Count Rating')
  
  # menampilkan wordcloud
  comment_words = ''
  stopwords = set(STOPWORDS)

  # iterate through the csv file
  for val in data.Reviews:
    
    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()
    
    # Converts each token into lowercase
    for i in range(len(tokens)):
      tokens[i] = tokens[i].lower()
    
    comment_words += " ".join(tokens)+" "

  wordcloud = WordCloud(width = 600, height = 600,
          background_color ='white',
          stopwords = stopwords,
          min_font_size = 10).generate(comment_words)

  # plot the WordCloud image					
  fig_wc = plt.figure(figsize=(10,5))
  ax = plt.axes()
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis("off")
  fig_wc.tight_layout(pad = 0)
  
  # mengatur kolom menjadi dua (kolom kanan dan kiri) untuk menampilkan ploting
  col1, col2 = st.columns(2)
  
  with col1:
    st.header('Total Jumlah per Rating')
    st.pyplot(fig_bar)
    
  with col2:
    st.header('WordCloud Reviews')
    st.pyplot(fig_wc)
else:
  st.write('Silahkan klik tombol scrape')

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)