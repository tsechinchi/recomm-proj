import os
import pandas as pd


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/movie_info.csv"
    df = pd.read_csv(path)
    df['genres'] = df.genres.str.split('|')

    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/genre.csv"
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index('id')
    return df


# user id, item id, rating, timestamp
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/ratings.csv"
    df = pd.read_csv(path, delimiter=",", header=0, names=["userId", "movieId", "rating", "timestamp"])
    # Convert epoch seconds to a datetime column so timeline-based loading/filters are possible
    df['rated_at'] = pd.to_datetime(df['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
    # Keep timestamp (epoch) and the converted datetime for flexibility
    df = df[['userId', 'movieId', 'rating', 'timestamp', 'rated_at']]

    return df


# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)