# Movie Recommendation System (Content-Based, TF-IDF)

A content-based movie recommender built on plot overviews, genres, and taglines. Given a movie title, it returns the N most similar movies using TF-IDF vectorization and cosine similarity.

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Metadata for ~45,000 movies scraped from TMDB and GroupLens, including overviews, genres, taglines, cast/crew, ratings, and revenue.

This project uses only `movies_metadata.csv`. The full Kaggle dataset also includes `credits.csv`, `keywords.csv`, `links.csv`, and `ratings.csv` (user-level ratings), which are **not** used here.

## How It Works

1. Load `movies_metadata.csv`, drop exact duplicate rows.
2. Keep `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`.
3. Parse the `genres` field (stringified list of dicts) into plain genre names.
4. Concatenate `overview + genres + tagline` into a single `tags` field per movie.
5. Clean `tags`: lowercase, strip punctuation, remove stopwords, lemmatize (NLTK).
6. Vectorize `tags` with `TfidfVectorizer` (unigrams + bigrams, max 50,000 features).
7. For a query movie, compute cosine similarity between its TF-IDF vector and all others; return the top-N closest.

## Setup

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
```

The notebook downloads required NLTK corpora (`stopwords`, `wordnet`) on first run.

Download `movies_metadata.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and place it in the project root.

## Usage

```python
recommend('The Godfather', n=10)
```

Returns the titles of the 10 movies most similar to the query, ranked by cosine similarity.


## Known Issues

- **Duplicate titles break title-based lookup.** 3,188 titles (about 7% of the dataset) appear more than once (e.g. "Hamlet" appears 9 times, "Cinderella" 11 times). The title-to-index map keeps only the first match, and querying certain duplicated titles raises an `IndexError` rather than returning a result.
- **`vote_average` and `popularity` are loaded but unused.** Recommendations are based purely on text similarity, so a high-quality match can rank below a near-empty, low-rated entry that happens to share more raw terms.
- **No persistence.** The TF-IDF matrix and vectorizer are rebuilt in-memory each run; there is no serialization step for reuse outside the notebook session.
- **Redundant text cleaning.** Stopwords are removed manually before lemmatization, then `TfidfVectorizer(stop_words='english')` removes stopwords again from already-cleaned text.

## Possible Improvements

- Re-rank top-N candidates by a `vote_count`-weighted score to reduce low-signal matches.
- Increase the relative weight of genre tokens in the tag string, or vectorize genres separately and blend similarity scores.
- Deduplicate or disambiguate titles before building the lookup index; key recommendations by `id` rather than `title`.
- Persist the fitted vectorizer and TF-IDF matrix (e.g. with `joblib`) for reuse without recomputing.

