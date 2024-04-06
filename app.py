from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)

# path to the dataset folder
data_folder = "data/"

# list all CSV files in the data directory
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# load all CSV files into a single DataFrame
all_movies_df = pd.concat(
    (pd.read_csv(f"{data_folder}{file}") for file in csv_files),
    ignore_index=True
)

# print columns to ensure the dataset loaded correctly
print(all_movies_df.columns)

@app.route('/')
def quiz():
    # Here, you can use all_movies_df as needed for your quiz functionality
    return render_template('quiz.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    preferred_genre = request.form.get('genre')  # Using .get for safer access
    minimum_rating = float(request.form.get('rating', 0))  # Default to 0 if not found

    # Filter the dataset for movies that match the preferences
    filtered_movies = all_movies_df[
        (all_movies_df['genre'].str.contains(preferred_genre, case=False, na=False)) &
        (all_movies_df['rating'] >= minimum_rating)
    ]

    # Select a random movie from the filtered dataset
    if not filtered_movies.empty:
        recommended_movie = filtered_movies.sample(1).iloc[0]
        return render_template('recommendation.html', 
                               title=recommended_movie['movie_name'],  # Adjusted to use the correct column name
                               genre=recommended_movie['genre'], 
                               rating=recommended_movie['rating'])
    else:
        return render_template('recommendation.html', 
                               title="Sorry, no movies found matching your criteria.",
                               genre="", 
                               rating="")

@app.route('/about')
def about():
    return render_template('about.html')  # Ensure you have an about.html template


if __name__ == '__main__':
    app.run(debug=True)
