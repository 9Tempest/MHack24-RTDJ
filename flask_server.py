from flask import Flask, jsonify

from multiprocessing import Manager
from spotify_api import getSimilarSong
import random
import time

# Initialize Flask app
app = Flask(__name__)

tracks = []

@app.route('/api/current-track')
def current_track():
    # tracks = [
    #     {"name": "Blinding Lights", "artist": "The Weeknd", "albumCover": "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36"},
    #     {"name": "Shape of You", "artist": "Ed Sheeran", "albumCover": "https://i.scdn.co/image/ab67616d0000b27a8b9c1de4456a42549baf23a7"},
    #     {"name": "Dance Monkey", "artist": "Tones and I", "albumCover": "https://i.scdn.co/image/ab67616d0000b273c6f7af36ecbc7615bad11a0b"}
    # ]
    return jsonify(tracks[0])

@app.route('/api/ai-thoughts')
def ai_thoughts():
    thoughts = [
        "Analyzing crowd energy levels...",
        "Detecting a shift in musical preferences...",
        "Evaluating dance floor engagement...",
        "Considering tempo changes for optimal flow..."
    ]
    return jsonify({"thoughts": random.choice(thoughts)})

@app.route('/api/crowd-mood')
def crowd_mood():
    moods = ["Energetic", "Relaxed", "Excited", "Mellow"]
    return jsonify({"mood": random.choice(moods)})

@app.route('/api/upcoming-tracks')
def upcoming_tracks():
    # tracks = [
    #     {"name": "Don't Start Now", "artist": "Dua Lipa"},
    #     {"name": "Watermelon Sugar", "artist": "Harry Styles"},
    #     {"name": "Levitating", "artist": "Dua Lipa"},
    #     {"name": "Blinding Lights", "artist": "The Weeknd"}
    # ]
    return jsonify({"tracks": tracks[1:]})

@app.route('/api/spotify-token')
def spotify_token():
    return jsonify({"token": "mock-spotify-token"})

def get_songs():
    for i in range(5):
        try:
            file = open("shared.txt","r")
            ans = [float(i) for i in file.read().split(',')[0:4]]
            #print(test)
            # Return the floats in a JSON response
            temp = [{"name" : i["track_name"], "artist" : "temp", "albumCover" : "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36"} for i in getSimilarSong(ans)]
            
            return temp
        except Exception as e:
            print(e)
            continue

kill_flag = False;
tracks = get_songs()
app.run(debug=True, threaded = True, port=5328)
        