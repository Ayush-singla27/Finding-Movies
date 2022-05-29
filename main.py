import pandas as pd
import streamlit as st
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


# ___FUNCTIONS___
# stemmer function
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# To find the common values in 2 lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# function to find imdbId for posters
def get_tmdb_id(lst, movies_list, links_list):
    movies_list = movies_list.merge(links_list, on="movieId", how="outer")
    movies_list.fillna(0)
    result1 = []
    for itr in lst:
        for j in range(len(movies_list['title'])):
            if itr == movies_list.loc[j]['title']:
                result1.append(movies_list.loc[j]['tmdbId'])
    return result1


# function to fetch posters from tmdb
# tmdb api :- https://www.themoviedb.org/documentation/api
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# collaborative filtering basics : https://developers.google.com/machine-learning/recommendation/collaborative/basics
def collaborative_rec(movie, movie_titles, rating_input):
    rating_input = pd.merge(rating_input, movie_titles, on='movieId')
    moviemat = rating_input.pivot_table(index='userId', columns='title', values='rating')
    ratings = pd.DataFrame(rating_input.groupby('title')['rating'].mean())
    ratings['num of ratings'] = pd.DataFrame(rating_input.groupby('title')['rating'].count())
    user_ratings = moviemat[movie]

    # creating correlation series/list
    similar_list = moviemat.corrwith(user_ratings)
    corr_df = pd.DataFrame(similar_list, columns=['Correlation'])
    corr_df.dropna(inplace=True)
    corr_df = corr_df.join(ratings['num of ratings'])

    # only movies with number of ratings > 100 are finalised to prevent recommending movies with less ratings
    rst = corr_df[corr_df['num of ratings'] > 100].sort_values('Correlation', ascending=False).index
    result = []
    for i in range(len(rst)):
        result.append(rst[i])
    return result


# content based filtering basics : https://developers.google.com/machine-learning/recommendation/content-based/basics
def content_based_rec(movie, movies, tags):
    movies = movies.merge(tags, on="movieId", how="outer")
    movies.tag = movies.tag.fillna('')
    movies['tag'] = movies['tag'] + " " + movies['genres']
    movies['tag'] = movies['tag'].str.replace('|', ' ', regex=True)
    movies = movies[['movieId', 'tag', 'title']]

    # converting tags into proper format for vectorisation
    movies['tag'] = movies.groupby(['movieId'])['tag'].transform(lambda x: ' '.join(str(v) for v in x))
    movies = movies.drop_duplicates()
    movies = movies.reset_index()
    movies['tag'] = movies['tag'].apply(lambda x: x.lower())
    movies['tag'] = movies['tag'].apply(stem)

    # applying vectorisation and cosine similarity from sklearn lib
    # to learn more : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tag']).toarray()
    similarity = cosine_similarity(vectors)
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    # order is lost during sorting, so we enumerate it to keep the index
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:101]
    result = []
    for i in movie_list:
        result.append(movies.iloc[i[0]].title)
    return result


# _______________________________________________________________________________________________________________


# input
movies = pd.read_csv('movies.csv')
links = pd.read_csv('links.csv')
rating_csv = pd.read_csv('ratings.csv')
tags_csv = pd.read_csv('tags.csv')

# UI using streamlit
# streamlit documentation : https://docs.streamlit.io/library/api-reference
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Microsoft Engage : Movie recommendation system ')
select_movie = st.selectbox('Movie Name here!', movies['title'].values, key='mix')

recommendation_collab = collaborative_rec(select_movie, movies, rating_csv)
recommendation_content = content_based_rec(select_movie, movies, tags_csv)

if st.button('Recommend'):

    result = intersection(recommendation_collab, recommendation_content)
    for i in range(5 - len(result)):
        result.append(recommendation_content[i])
    tmdb_id = get_tmdb_id(result, movies, links)

    recommended_movie_posters = []
    for movie_id in tmdb_id:
        recommended_movie_posters.append(fetch_poster(movie_id))

    # OUTPUT
    st.markdown("<h1 style='text-align: center; color: #a0fbff;'>Our Recommendations</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(result[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(result[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(result[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(result[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(result[4])
        st.image(recommended_movie_posters[4])

    col1, col2 = st.columns(2)
    with col1:
        st.write('______________________________________________')
        st.subheader('Movies loved by similar users')
        for i in range(5):
            if i >= len(recommendation_collab):
                st.write('----  lack of user ratings  ----')
                break
            st.write(recommendation_collab[i])
    with col2:
        st.write('______________________________________________')
        st.subheader('Movies with similar content')
        for i in range(5):
            st.write(recommendation_content[i])
