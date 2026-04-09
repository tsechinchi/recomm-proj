import os
import pandas as pd


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    root_path = os.path.abspath(os.getcwd())
    ml_data_dir = f"{root_path}/flaskr/static/ml_data"

    # Prefer MovieLens 1M when available (ml-1m/movies.dat).
    ml1m_movies_path = f"{ml_data_dir}/ml-1m/movies.dat"
    if os.path.exists(ml1m_movies_path):
        df = pd.read_csv(
            ml1m_movies_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        extracted_year = df["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
        clean_title = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

        df["year"] = pd.to_numeric(extracted_year, errors="coerce")
        df["title"] = clean_title
        df["overview"] = ""
        df["cover_url"] = ""
        df["genres"] = df["genres"].fillna("(no genres listed)").str.split("|")
        # Keep a UI-friendly alias used in some templates.
        df["release_date"] = df["year"]
        return df[["movieId", "title", "genres", "year", "overview", "cover_url", "release_date"]]

    # Backward-compatible fallback to existing CSV pipeline.
    csv_path = f"{ml_data_dir}/movie_info.csv"
    df = pd.read_csv(csv_path)
    if "genres" in df:
        df["genres"] = df["genres"].fillna("(no genres listed)").str.split("|")
    else:
        df["genres"] = [["(no genres listed)"]] * len(df)
    if "year" not in df:
        df["year"] = pd.to_numeric(
            df.get("title", pd.Series([""] * len(df))).astype(str).str.extract(r"\((\d{4})\)\s*$", expand=False),
            errors="coerce",
        )
    if "overview" not in df:
        df["overview"] = ""
    if "cover_url" not in df:
        df["cover_url"] = ""
    if "release_date" not in df:
        df["release_date"] = df["year"]
    return df


# A list of the genres.
def getGenre():
    root_path = os.path.abspath(os.getcwd())
    ml_data_dir = f"{root_path}/flaskr/static/ml_data"

    # Prefer MovieLens 1M genres inferred from movies.dat.
    ml1m_movies_path = f"{ml_data_dir}/ml-1m/movies.dat"
    if os.path.exists(ml1m_movies_path):
        movies_df = pd.read_csv(
            ml1m_movies_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        all_genres = set()
        for genre_blob in movies_df["genres"].fillna("(no genres listed)").tolist():
            for genre in str(genre_blob).split("|"):
                all_genres.add(genre)
        ordered = sorted(all_genres)
        return pd.DataFrame({"name": ordered, "id": list(range(1, len(ordered) + 1))})

    # Backward-compatible fallback.
    path = f"{ml_data_dir}/genre.csv"
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index("id")
    return df


# user id, item id, rating, timestamp
def getRates():
    root_path = os.path.abspath(os.getcwd())
    ml_data_dir = f"{root_path}/flaskr/static/ml_data"

    # Prefer MovieLens 1M ratings.dat when available.
    ml1m_ratings_path = f"{ml_data_dir}/ml-1m/ratings.dat"
    if os.path.exists(ml1m_ratings_path):
        df = pd.read_csv(
            ml1m_ratings_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
    else:
        path = f"{ml_data_dir}/ratings.csv"
        df = pd.read_csv(path, delimiter=",", header=0, names=["userId", "movieId", "rating", "timestamp"])

    # Convert epoch seconds to a datetime column so timeline-based loading/filters are possible
    df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    # Keep timestamp (epoch) and the converted datetime for flexibility
    df = df[["userId", "movieId", "rating", "timestamp", "rated_at"]]

    return df


# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []
    timestamp = []

    now_epoch = int(pd.Timestamp.utcnow().timestamp())
    for index, rate in enumerate(rates):
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(float(items[2]))
        if len(items) >= 4 and str(items[3]).strip():
            try:
                timestamp.append(int(float(items[3])))
            except ValueError:
                timestamp.append(now_epoch + index)
        else:
            timestamp.append(now_epoch + index)

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
        "timestamp": timestamp,
    }

    return pd.DataFrame(ratings_dict)
