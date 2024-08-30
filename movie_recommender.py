import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import difflib
import requests
import gzip
import shutil

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to download and preprocess data
@st.cache_data
def load_and_preprocess_data():
    if os.path.exists('preprocessed_movie_data.parquet'):
        return pd.read_parquet('preprocessed_movie_data.parquet')
    
    # URLs for IMDb datasets
    basics_url = "https://datasets.imdbws.com/title.basics.tsv.gz"
    ratings_url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
    
    # Download and extract files
    for url in [basics_url, ratings_url]:
        filename = url.split('/')[-1]
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        with gzip.open(filename, 'rb') as f_in:
            with open(filename[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filename)  # Remove the .gz file
    
    basics = pd.read_csv('title.basics.tsv', sep='\t', dtype=str)
    ratings = pd.read_csv('title.ratings.tsv', sep='\t', dtype=str)
    
    # Remove the downloaded TSV files to save space
    os.remove('title.basics.tsv')
    os.remove('title.ratings.tsv')
    
    movies = basics[basics['titleType'] == 'movie']
    movie_data = pd.merge(movies, ratings, on='tconst', how='inner')
    
    movie_data['startYear'] = pd.to_numeric(movie_data['startYear'], errors='coerce')
    movie_data['averageRating'] = pd.to_numeric(movie_data['averageRating'], errors='coerce')
    movie_data['numVotes'] = pd.to_numeric(movie_data['numVotes'], errors='coerce')
    
    movie_data = movie_data[(movie_data['startYear'] >= 1990) & (movie_data['numVotes'] >= 1000)]
    
    # Save as compressed parquet file instead of pickle
    movie_data.to_parquet('preprocessed_movie_data.parquet', compression='gzip')
    return movie_data

# Load the data
movie_data = load_and_preprocess_data()
all_movies = sorted(movie_data['primaryTitle'].unique())
all_genres = sorted(set(','.join(movie_data['genres'].dropna()).split(',')))

st.title('Film Öneri Sistemi')

st.header('Nasıl Çalışır?')
st.markdown("""
- Sevdiğiniz film türlerini seçin
- En sevdiğiniz 3 filmi belirtin
- 'Öneriler Al' butonuna tıklayın
- Daha fazla öneri için 'Başka Öner' butonuna tıklayın!
""")

# User input for genres
user_genres = st.multiselect('Sevdiğiniz film türlerini seçin:', all_genres)

# User input for favorite movies
st.subheader('En Sevdiğiniz 3 Film')
user_movies = [st.selectbox(f'{i+1}. Film', [''] + all_movies, key=f'movie_{i}') for i in range(3)]
user_movies = [movie for movie in user_movies if movie]  # Remove empty selections

def get_recommendations(user_movies, user_genres, excluded_movies):
    if not user_movies and not user_genres:
        return pd.DataFrame()

    # Prepare the messages for ChatGPT
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides movie recommendations."},
        {"role": "user", "content": f"I enjoy movies like {', '.join(user_movies)} and I like {', '.join(user_genres)} genres. Can you recommend 10 movies for me? Please exclude these movies: {', '.join(excluded_movies)}. Provide only the movie titles, one per line, without numbering."}
    ]

    # Use the OpenAI API to get recommendations from ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200
    )

    # Parse the response from ChatGPT
    content = response.choices[0].message['content'].strip()
    recommended_titles = [title.strip() for title in content.split('\n') if title.strip()]

    # Use difflib to find the closest matches in the movie_data
    all_titles = movie_data['primaryTitle'].unique()
    matched_movies = []
    for title in recommended_titles:
        matches = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.6)
        if matches:
            matched_title = matches[0]
            matched_movie = movie_data[movie_data['primaryTitle'] == matched_title].iloc[0]
            matched_movies.append(matched_movie)

    # Convert the matched movies to a DataFrame
    recommended_movies = pd.DataFrame(matched_movies)

    # If no matching titles found in movie_data, return an empty DataFrame
    if recommended_movies.empty:
        return pd.DataFrame()

    # Exclude movies that are in the excluded_movies list
    recommended_movies = recommended_movies[~recommended_movies['primaryTitle'].isin(excluded_movies)]

    # Sort by rating and return top 5
    return recommended_movies.sort_values('averageRating', ascending=False).head(5)

if 'excluded_movies' not in st.session_state:
    st.session_state.excluded_movies = set()

if st.button('Öneriler Al') or st.session_state.get('show_recommendations', False):
    st.session_state.show_recommendations = True
    recommended_movies = get_recommendations(user_movies, user_genres, st.session_state.excluded_movies)

    if not recommended_movies.empty:
        st.subheader('Önerilen Filmler')
        recommendations_df = pd.DataFrame({
            'Film Adı': recommended_movies['primaryTitle'] + ' (' + recommended_movies['startYear'].astype(str) + ')',
            'Tür': recommended_movies['genres'],
            'Puan': recommended_movies['averageRating'].round(1)
        })
        
        # Style the dataframe
        styled_df = recommendations_df.style.set_properties(**{
            'background-color': 'lightblue',
            'color': 'black',
            'border-color': 'white'
        })
        
        # Display the styled dataframe
        st.dataframe(styled_df, use_container_width=True)
        
        st.session_state.excluded_movies.update(recommended_movies['primaryTitle'])
        
        if st.button('Başka Öner'):
            st.session_state.show_recommendations = True
            st.rerun()
    else:
        st.warning("Üzgünüz, kriterlerinize uygun film önerisi bulunamadı.")
else:
    st.session_state.show_recommendations = False