# Welcome to Movie-Match!
This Flask application serves as a movie recommendation system, leveraging a machine learning model built with PyTorch to provide movie suggestions based on user preference. This way every user should find a movie that fits. 

## Features
- **Basic Movie Recommendations**: Allows users to select their preferred genre and minimum rating to receive movie recommendations.
- **Ultimate Predictor**: A machine learning-based feature that provides movie recommendations based on a favorite movie.

## Requirements
- Python 3.6+
- Flask
- Pandas
- NumPy
- PyTorch
- Dataset: The application uses a specific dataset of movies for recommendations. Download the dataset in CSV format from the following Kaggle link and place it in the data/ directory within the application's folder. [IMDb Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre?select=fantasy.csv)

Ensure all requirements are met to avoid any issues with running the application.


## Installation
1. Clone the repository to your local machine.
```bash
git clone <repository-url>
```
2. Navigate to the cloned directory.
```bash
cd <repository-name>
```
3. Install the required Python packages.
```bash
pip install Flask pandas numpy torch
```
4. Place your dataset(s) of movies in the data/ directory. Ensure each dataset is in CSV format.


## Usage
1. Start the Flask application.
```bash
python app.py
```
2. Open your web browser and navigate to http://127.0.0.1:5000/ to access the application (private window if you wish to contribute).
3. Follow the on-screen instructions to get movie recommendations or use the ultimate predictor feature.

## Contributing
Contributions to improve the application are welcome. Please follow these steps to contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature/AmazingFeature).
- Commit your changes (git commit -m 'Add some AmazingFeature').
- Push to the branch (git push origin feature/AmazingFeature).
- Open a pull request.


## Acknowledgements
- This project uses data from [IMDb Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre?select=fantasy.csv).
- Machine learning model powered by PyTorch.
