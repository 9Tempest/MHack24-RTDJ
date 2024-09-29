# Python Script to Find Top 5 Most Similar Songs Based on Audio Features, Considering Recommendation Score and Uniqueness

import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import webbrowser

# Step 1: Set up MongoDB connection
CLIENT_SECRET = "c615959b63d04e8f959b01800bf7ef4b"
CLIENT_ID = '29e011cdf67041baa13d003873608c04'
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

    # Ensure 'recommendation_score' column exists, if not create and set default to 0
    if 'recommendation_score' not in df.columns:
        df['recommendation_score'] = 0  # Assign default value of 0 to missing recommendation scores
    
    # Step 3: Define the input song features for similarity search
    input_song = {
        "track_name": "Your Input Track Name",
        "danceability": embeedings[0],
        "energy": embeedings[1],
        "tempo": embeedings[2],
        "valence": embeedings[3]
    }

    # Step 4: Create a DataFrame for the input song and existing songs
    input_features = np.array([[input_song['danceability'], input_song['energy'], input_song['tempo'], input_song['valence']]])
    existing_features = df[['danceability', 'energy', 'tempo', 'valence']].values

    # Step 5: Calculate cosine similarity between the input song and existing songs
    similarity_scores = cosine_similarity(input_features, existing_features)[0]

    # Step 6: Filter out perfect matches (similarity score of 1.0)
    non_perfect_match_indices = np.where(similarity_scores < 1.0)[0]

    # Step 7: Adjust similarity based on recommendation score
    recommendation_scores = df.loc[non_perfect_match_indices, 'recommendation_score'].fillna(0).values  # Default score of 0 if not found
    adjusted_similarity_scores = similarity_scores[non_perfect_match_indices] / (1 + recommendation_scores)  # Penalize songs with higher recommendation counts

    # Step 8: Sort the songs by adjusted similarity score
    sorted_indices = non_perfect_match_indices[np.argsort(adjusted_similarity_scores)[::-1]]

    # Step 9: Retrieve the most similar songs' details, ensuring uniqueness
    unique_songs = set()
    top_songs_data = []
    for idx in sorted_indices:
        song = df.iloc[idx]
        track_id = song['track_id']

        if track_id not in unique_songs:
            unique_songs.add(track_id)
            top_songs_data.append(song)
            if len(top_songs_data) >= 5:  # Stop once we have 5 unique songs
                break

    # Step 10: Authenticate with Spotify using Spotipy
    sp = Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri="http://localhost:8888/callback",
        scope="user-modify-playback-state,user-read-playback-state"
    ))

    # Step 11: Print out the details of the top 5 unique most similar songs
    print(embeedings)
    print("Top 5 Most Similar Unique Songs (Considering Recommendation Score):")

    for song in top_songs_data:
        track_uri = song['track_id']  # Assuming 'track_id' is the Spotify URI for the song
        print(f"Track ID: {song['track_id']}")
        print(f"Track Name: {song['track_name']}")
        print(f"Danceability: {song['danceability']}")
        print(f"Energy: {song['energy']}")
        print(f"Tempo: {song['tempo']}")
        print(f"Valence: {song['valence']}")
        print(f"track_uri: {track_uri}")

        # Step 12: Update the recommendation score for each song in the database
        collection.update_one(
            {'track_id': song['track_id']},
            {'$inc': {'recommendation_score': 1}}  # Increment recommendation score by 1
        )

    # Return the URIs of the top 5 most similar unique songs
    return [song['track_id'] for song in top_songs_data]

# Example usage
if __name__ == '__main__':
    print(getSimilarSong([0.7, 0.8, 120.0, 0.5])) #Add in the model weights here.
