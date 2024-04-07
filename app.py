from flask import Flask, render_template, request, redirect, url_for, flash

import os
import predict
import pandas as pd    
import numpy as np

# ----- flask application -------
app = Flask(__name__) # initialize flask application

data_folder = "data/" # path to the dataset folder

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')] # list all CSV files in data direcory

# load all CSV files into a single DataFrame
all_movies_df = pd.concat(
    (pd.read_csv(f"{data_folder}{file}") for file in csv_files),
    ignore_index=True
)

predictor = predict.Predictor(10000)

@app.route('/') # app route for home where the basic quiz takes place
def quiz():
    return render_template('quiz.html')

@app.route('/recommend', methods=['POST']) # app route for the results of the basic quiz
def recommend():
    preferred_genre = request.form.get('genre')  # get genre as user input
    minimum_rating = float(request.form.get('rating', 0)) # get minimum rating as user input

    # filter the dataset for movies that match the preferences
    filtered_movies = all_movies_df[
        (all_movies_df['genre'].str.contains(preferred_genre, case=False, na=False)) &
        (all_movies_df['rating'] >= minimum_rating)
    ]

    # check if any movies were found
    if not filtered_movies.empty:
        # convert DataFrame to a list of dictionaries for easier handling in the template
        movies_list = filtered_movies.to_dict('records')
        return render_template('recommendation.html', movies=movies_list)
    else:
        # pass an empty list if no movies were found
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


@app.route('/check-movie', methods=['POST']) # app route for check-movie
def check_movie():
    movie_title = request.json['movie_title'].lower() 
    movie_exists = movie_title in all_movies_df['movie_name'].str.lower().unique().tolist()
    return {'exists': movie_exists}


# ----- start the flask application ------
if __name__ == '__main__':
    app.run(debug=True)
