# Welcome to Movie-Match!
This Flask application serves as a movie recommendation system that also leverages a machine learning model built with PyTorch to provide movie suggestions based on a user's preferred movie. We hope that this system will help every user find a movie that fits their interests.

## Features
- **Basic Movie Recommendation**: Allows users to select their preferred genre and minimum rating to receive movie recommendations generated through basic filtering.
- **Ultimate Movie Prediction**: A machine learning based feature that provides a prediction of movies that the user may like based on the user's favorite movie.

## Requirements
- Python 3.6+
- Flask
- Pandas
- NumPy
- Torch
- Dataset: The application uses a specific dataset of movies for recommendations. Download the dataset from the following Kaggle link and place all CSV files in a directory called "data" at the root of the projects directory. [IMDb Movie Dataset: All Movies by Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre?select=fantasy.csv)

## Installation
1. Clone the repository to your local machine.
```bash
git clone <repository-url>
```
2. Navigate to the cloned directory.
```bash
cd <repository-name>
```
3. Make sure all requirements are installed.
```bash
pip install -r requirements.txt
```
4. Place your dataset of movies in a directory named "data" at the root of the projects directory (ensure each dataset file is in CSV format)


## Usage
**There is the optional possibility to run it in a virtual environment. To do so on a mac, enter the following commands in your terminal:**

1. Create a Virtual Environment
```bash
  /opt/homebrew/bin/python3 -m venv /Users/pathtoproject/Movie-Match/venv
```
2. Activate the Virtual Environment
```bash
  source /Users/pathtoproject/Movie-Match/venv/bin/activate
```
3. Install Dependencies
```bash
  pip install -r requirements.txt
```
4. Run the Application
```bash
  python app.py
```
5. Open your web browser and navigate to http://127.0.0.1:5000/ to access the application (recommended to open in a private window - especially in for contribution).
6. Follow the on-screen instructions to get movie recommendations or use the ultimate predictor feature.

**If you want to run it without creating a virtual enviromnent, simply follow continue with this step:**

1. Start the Flask application.
```bash
python app.py
```
2. Open your web browser and navigate to http://127.0.0.1:5000/ to access the application (recommended to open in a private window - especially in for contribution).
3. Follow the on-screen instructions to get movie recommendations or use the ultimate predictor feature.

## Contributing
Contributions to improve the application are welcome. Please follow these steps to contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature/AmazingFeature).
- Commit your changes (git commit -m 'Add some AmazingFeature').
- Push to the branch (git push origin feature/AmazingFeature).
- Open a pull request.


## Acknowledgements
- This project uses data from [IMDb Movie Dataset: All Movies by Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre?select=fantasy.csv).
- Machine learning model powered by PyTorch.
