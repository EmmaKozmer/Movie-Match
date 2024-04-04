from flask import Flask, render_template
import os
import pandas as pd

app = Flask(__name__)

# Path to the dataset folder
data_folder = "data/"

# List all CSV files in the data directory
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Load all CSV files into a single DataFrame
all_movies_df = pd.concat(
    (pd.read_csv(f"{data_folder}{file}") for file in csv_files),
    ignore_index=True
)

# You can print the columns to verify it's loaded correctly
# This print statement will execute when you start your Flask app
print(all_movies_df.columns)

@app.route('/')
def quiz():
    # Here, you can use all_movies_df as needed for your quiz functionality
    return render_template('quiz.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    preferred_genre = request.form['genre']
    minimum_rating = float(request.form['rating'])

    # Filter the dataset for movies that match the preferences
    filtered_movies = all_movies_df[
        (all_movies_df['genre'].str.contains(preferred_genre, case=False)) &
        (all_movies_df['rating'] >= minimum_rating)
    ]
    
    # Select a random movie from the filtered dataset
    if not filtered_movies.empty:
        recommended_movie = filtered_movies.sample(1).iloc[0]
        movie_title = recommended_movie['title']
        movie_genre = recommended_movie['genre']
        movie_rating = recommended_movie['rating']
        return render_template('recommendation.html', title=movie_title, genre=movie_genre, rating=movie_rating)
    else:
        return "Sorry, no movies found matching your criteria. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
