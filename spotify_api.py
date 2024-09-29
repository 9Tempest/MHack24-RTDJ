# Python Script to Find Most Similar Song Based on Audio Features:

import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import webbrowser

# Step 1: Set up MongoDB connection
# Step 1: Load the client secret and Client ID
CLIENT_SECRET = "c615959b63d04e8f959b01800bf7ef4b"
CLIENT_ID = '29e011cdf67041baa13d003873608c04'
# Step 2: Authenticate with Spotify API using OAuth
MONGO_URI = 'mongodb+srv://talluriv:EAgiICP8cSA4btnR@cluster0.8sbfv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
def getSimilarSong(embeedings):
    # Step 1: Load the client secret and Client ID
    client = MongoClient(MONGO_URI)

    # Access the database and collection
    db = client['spotify_db']
    collection = db['audio_features']

    # Step 2: Load data from MongoDB into a DataFrame
    audio_features_list = list(collection.find({}))
    df = pd.DataFrame(audio_features_list)

    # Step 3: Define the input song features for similarity search (NEED TO CHANGE THIS FOR VECTOR EMBEDDINGS)
    input_song = {
        "track_name": "Your Input Track Name",  # Change this to the song name you are interested in
        "danceability": embeedings[0],  # Change this to the desired danceability score
        "energy": embeedings[1],  # Change this to the desired energy score
        "tempo": embeedings[2],  # Change this to the desired tempo score
        "valence": embeedings[3]  # Change this to the desired valence score
    }

    # Step 4: Create a DataFrame for the input song and existing songs
    input_features = np.array([[input_song['danceability'], input_song['energy'], input_song['tempo'], input_song['valence']]])
    existing_features = df[['danceability', 'energy', 'tempo', 'valence']].values

    # Step 5: Calculate cosine similarity between the input song and existing songs
    similarity_scores = cosine_similarity(input_features, existing_features)

    # Step 6: Find the index of the most similar song
    most_similar_index = np.argmax(similarity_scores)

    # Step 7: Retrieve the most similar song details
    most_similar_song = df.iloc[most_similar_index]

    # Step 8: Authenticate with Spotify using Spotipy
    sp = Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,        # Replace with your Spotify client ID
        client_secret=CLIENT_SECRET,# Replace with your Spotify client secret
        redirect_uri="http://localhost:8888/callback",  # Replace with your redirect URI
        scope="user-modify-playback-state,user-read-playback-state"  # Scopes required to control playback
    ))

    # Get the track URI
    track_uri = most_similar_song['track_id']  # Assuming 'track_id' is the Spotify URI for the song

    # Step 8: Print out the most similar song details
    print(embeedings)
    
    print("Most Similar Song:")
    print(f"Track ID: {most_similar_song['track_id']}")
    print(f"Track Name: {most_similar_song['track_name']}")
    print(f"Danceability: {most_similar_song['danceability']}")
    print(f"Energy: {most_similar_song['energy']}")
    print(f"Tempo: {most_similar_song['tempo']}")
    print(f"Valence: {most_similar_song['valence']}")
    print(f"track_uri: {most_similar_song}")

    # Play the most similar song (must have an active Spotify playback device)
    #real_url = 'https://open.spotify.com/track/' + track_uri
    #webbrowser.open(real_url)
    return track_uri

# Example usage
    # input_song = {
    #     "danceability": 0.7,  # Change this to the desired danceability score
    #     "energy": 0.8,  # Change this to the desired energy score
    #     "tempo": 120.0,  # Change this to the desired tempo score
    #     "valence": 0.5  # Change this to the desired valence score
    # }
if __name__ == '__main__':
    print(getSimilarSong([0.7, 0.8, 120.0, 0.5]))