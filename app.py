import pickle
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# LOAD DATA & COMPUTE SIMILARITY (cached so it only runs once per session)
@st.cache_data
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity


# FETCH POSTER
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=08b7405c5846b6595981b25f33687cb7&language=en-US".format(movie_id)
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/300x450?text=No+Poster"
    except Exception as e:
        print(e)
        return "https://via.placeholder.com/300x450?text=Error"


# RECOMMEND FUNCTION
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters


# UI
st.header('Movie Recommender System')

movies, similarity = load_data()

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


# BUTTON
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])

    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])

    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])

    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
