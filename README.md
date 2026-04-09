## Create an environment

```
conda create -n lab3
conda activate lab3

```

## Install Python packages 

```
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt

```
## alternative -uv
## Install uv

First, install uv (a fast Python package manager):

```bash
# On macOS/Linux with brew
brew install uv

# Or install via pip
pip install uv

# Or download from https://github.com/astral-sh/uv
## Run the project
# Sync dependencies from pyproject.toml
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows 

```
uv run flask --app flaskr run --debug # if you use non uv, remove uv run
```

## Add the recommendation algorithm
You only need to modify the `main.py` file. Its path is as follows:
```
path: /flaskr/main.py
```

## About the Dataset
The dataset path is: ./flaskr/static/ml_data/

The ratings.csv file includes the following columns:
- userId: the IDs of users.  
- movieId: the IDs of movies.  
- rating: the rating given by the user to the movie, on a 5-star scale
- timestamp: the time when the user rated the movie, recorded in seconds since the epoch (as returned by time(2) function). A larger timestamp means the rating was made later.

You can use pandas to convert the timestamp to standard date and time. For example, 1717665888 corresponds to 2024-06-06 09:24:48.
```
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)
```

## MovieLens 1M (Recommended for denser data)

The data loader now auto-detects MovieLens 1M if these files exist:

- `./flaskr/static/ml_data/ml-1m/movies.dat`
- `./flaskr/static/ml_data/ml-1m/ratings.dat`

When present, the app uses MovieLens 1M directly; otherwise it falls back to the legacy CSV files in `./flaskr/static/ml_data/`.

Expected source: MovieLens 1M from GroupLens.
