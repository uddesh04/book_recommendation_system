from operator import ipow
from requests import options
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from itertools import zip_longest
import urllib.request
import json
import textwrap
import base64


books=pd.read_csv("book dataset/BX-Books.csv", sep=";", encoding="latin-1", error_bad_lines=False)
users=pd.read_csv("book dataset/BX-Users.csv", sep=";", encoding="latin-1", error_bad_lines=False)
ratings=pd.read_csv("book dataset/BX-Book-Ratings.csv", sep=";", encoding="latin-1", error_bad_lines=False)
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-S','Image-URL-M','Image-URL-L']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher', 'Image-URL-S':'urls', 'Image-URL-M':'urlm', 'Image-URL-L':'urll'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)
x = ratings['user_id'].value_counts() > 200
y = x[x].index  #user_ids
ratings = ratings[ratings['user_id'].isin(y)]
rating_with_books = ratings.merge(books, on='ISBN')
number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating.shape
final_rating = final_rating[final_rating['number_of_ratings'] >= 30]
final_rating.drop_duplicates(['user_id','title'], inplace=True)
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)
book_sparse = csr_matrix(book_pivot)
book_list=[]
for i in range(len(book_pivot.index)):
    book_list.append(book_pivot.index[i])

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def make_prediction(bookname):

    model = NearestNeighbors(algorithm='brute')
    model.fit(book_sparse)
    user_input=-1
    for i in book_pivot.index:
        user_input=user_input+1
        if(i==bookname):
            break
    distances, suggestions = model.kneighbors(book_pivot.iloc[user_input, :].values.reshape(1, -1))
    listed=[]
    links=[]
    isbn=[]

    for i in suggestions:
        listed.append(book_pivot.index[i])
    for i in range(len(listed[0])):
        links.append(books.loc[books['title']==listed[0][i], 'urlm'])
        isbn.append(books.loc[books['title']==listed[0][i], 'ISBN'])
    links=list(list(zip(*links))[0])
    isbn=list(list(zip(*isbn))[0])
    res = {mk: (yr, md) for mk, yr, md in zip_longest(listed[0], links, isbn)}
    return res

@st.cache(suppress_st_warning=True,allow_output_mutation=True,hash_funcs={'_json.Scanner': hash})
def extract_bookinfo(ibsnn):
    base_api_link = "https://www.googleapis.com/books/v1/volumes?q=isbn:"
    user_input = ibsnn

    with urllib.request.urlopen(base_api_link + user_input) as f:
        text = f.read()
    decoded_text = text.decode("utf-8")
    obj = json.loads(decoded_text)
    volume_info = obj["items"][0] 
    authors = obj["items"][0]["volumeInfo"]["authors"]
    
    return volume_info,authors


def add_bg_from_url(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image:  url(data:image/{"png"};base64,{encoded_string.decode()});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url('bookbg.png') 

# Set the app title

st.title("Book Recommendation App")
st.write("A simple machine learning app to recommend you a book you may like")
form = st.form(key="my_form")
book_name = form.selectbox(label="Enter the name of the book you read recently", options=book_list)
submit = form.form_submit_button(label="Suggest something")
dic={}
if submit:
    # make prediction from the input text
    dic= make_prediction(book_name)
    l_list=list(dic.keys())

    # Display results of the function task
    st.header("Results")
    st.write("Here are some books you may like")
    
    for i in range(5):
        st.write("\n")
        col1, mid, col2 = st.columns([1,4,20])
        with col1:
            link_text=(dic[l_list[i]][0])
            st.image(link_text,width=120)
        with col2:
            linker="https://books.google.com/books?vid=ISBN"+str(dic[l_list[i]][1])
            st.subheader(l_list[i],anchor=linker)
            vi,aut=extract_bookinfo(dic[l_list[i]][1])
            genre=vi["volumeInfo"]["categories"]
            st.write("Author(s):", ",".join(aut), "||     Genre:", genre[0] )
            st.write(textwrap.fill(vi["searchInfo"]["textSnippet"], width=60))
            
        
