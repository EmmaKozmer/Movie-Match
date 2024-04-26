
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from src.data_loader import load_movie_data 
from src.predict import Predictor

import numpy as np

# ----- flask application -------
app = Flask(__name__) # initialize flask application

# path to the dataset folder and model path
data_folder = "data/"
local_model_path = "models/movie_predictor_model.pth"

# load all movie data in a DataFrame
all_movies_df = load_movie_data(data_folder)

# initialize the predictor for the local filesystem model - uncomment to use the small (10000) model for the ultimate predictor
#local_model_path = "models/movie_predictor_model.pth"
#predictor = Predictor(local_model_path, 10000)  # Assuming 10,000 is the number of movie IDs for the local model

# initialize the predictor for the Hugging Face model - uncomment to use the large (368300) model for the ultimate predictor
huggingface_model_identifier = 'Emm180/movie_match_model'
predictor = Predictor(huggingface_model_identifier, 368300)  # Assuming 368,000 is the number of movie IDs for the Hugging Face model*

# ----- app routes ------
@app.route('/')
def basic_quiz():
    return render_template('basic_quiz.html')


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
    # Check if any movies were found
    if not filtered_movies.empty:
        recommended_movie = filtered_movies.sample(1).iloc[0]
        return render_template('recommendation.html', 
                               title=recommended_movie['movie_name'],  # Adjusted to use the correct column name
                               genre=recommended_movie['genre'], 
                               rating=recommended_movie['rating'])
        # Convert DataFrame to a list of dictionaries for easier handling in the template
        movies_list = filtered_movies.to_dict('records')
        return render_template('recommendation.html', movies=movies_list)
    else:
        return render_template('recommendation.html', 
                               title="Sorry, no movies found matching your criteria.",
                               genre="", 
                               rating="")
        # Pass an empty list if no movies were found
        return render_template('recommendation.html', movies=[])


@app.route('/about') # app route for the results of the basic quiz
def about():
    return render_template('about.html')


@app.route('/ultimate-predictor') # app route for the ultimate predictor
def ultimatepredictor():
    return render_template('ultimate-predictor.html') 


@app.route('/predict', methods=['POST']) # app route for the results of the ultimate predictor
def predict():
    movie_title = request.form['movie_title'].strip().lower() 
    movie_titles = all_movies_df['movie_name'].str.lower().unique().tolist() # list of movies for the dataframe
    
    # check if movie is in movie title
    if movie_title not in movie_titles:
        flash("Sorry, we don't know about this movie.")
        return redirect(url_for('ultimatepredictor'))

    # define the matched movie
    matched_movie = all_movies_df[all_movies_df['movie_name'].str.lower() == movie_title]

    # define the preferred movie
    preferred_movie_id = np.where(all_movies_df['movie_name'].str.lower() == movie_title)[0][0]

    # check if a matching movie can be found
    if matched_movie.empty:
        flash("Sorry, we couldn't find a matching movie :(")
        return redirect(url_for('ultimatepredictor'))

    # define the number of predicted movies
    n = 5 

    # define the reommended movies
    recommended_movie_ids = predictor.predict(preferred_movie_id, n)
    recommended_movies = all_movies_df.iloc[recommended_movie_ids]

    list_of_dicts = recommended_movies.to_dict('records') # convert recommendet movies to dict

    return render_template('prediction_result.html', movies=list_of_dicts)

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/check-movie', methods=['POST'])
def check_movie():
    movie_title = request.json['movie_title'].lower()
    movie_exists = movie_title in all_movies_df['movie_name'].str.lower().unique().tolist()
    return jsonify(exists=movie_exists)




# ----- start the flask application ------
if __name__ == '__main__':
    app.run(debug=True)
