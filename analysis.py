import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

df = pd.read_csv('netflix_titles.csv')
df.shape


# 1. Data Cleaning

df.info()

# Handle missing values
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Parse date_added
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

df.isna().sum()


# 2. Exploratory Data Analysis


type_counts = df['type'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
        colors=['#E50914', '#221f1f'], startangle=90,
        wedgeprops={'edgecolor':'white'})
plt.title('Movies vs TV Shows on Netflix')
plt.show()


plt.figure(figsize=(12,5))
sns.countplot(data=df, x='year_added', hue='type',
              palette={'Movie':'#E50914','TV Show':'#221f1f'})
plt.title('Content Added to Netflix by Year')
plt.xlabel('Year Added')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


top_countries = df['country'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r')
plt.title('Top 10 Countries by Content Volume')
plt.xlabel('Number of Titles')
plt.show()


genre_series = df['listed_in'].str.split(', ').explode()
top_genres = genre_series.value_counts().head(15)
plt.figure(figsize=(10,6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='Reds_r')
plt.title('Top 15 Genres')
plt.xlabel('Number of Titles')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df['release_year'], bins=25, kde=False, color='#E50914')
plt.title('Distribution of Release Years')
plt.xlabel('Release Year')
plt.show()


movie_durations = df[df['type']=='Movie']['duration'].str.replace(' min','').astype(float)
plt.figure(figsize=(10,5))
sns.histplot(movie_durations, bins=30, color='#221f1f')
plt.axvline(movie_durations.mean(), color='#E50914', linestyle='--',
            label=f'Mean: {movie_durations.mean():.0f} min')
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.legend()
plt.show()


rating_order = df['rating'].value_counts().index
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='rating', order=rating_order, palette='Reds_r')
plt.title('Content Ratings Breakdown')
plt.xticks(rotation=45)
plt.show()


# 3. Content-Based Recommendation System


df['content_soup'] = (
    df['listed_in'].fillna('') + ' ' +
    df['cast'].fillna('') + ' ' +
    df['director'].fillna('') + ' ' +
    df['description'].fillna('')
)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content_soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_index = pd.Series(df.index, index=df['title'])
title_to_index = title_to_index[~title_to_index.index.duplicated(keep='first')]

def recommend(title, n=10):
    if title not in title_to_index:
        matches = df[df['title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            return f"No title found matching '{title}'"
        idx = matches.index[0]
    else:
        idx = title_to_index[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    rec_indices = [i for i, score in sim_scores]

    result = df.loc[rec_indices, ['title', 'type', 'listed_in', 'release_year']].copy()
    result['similarity'] = [round(score, 3) for _, score in sim_scores]
    return result.reset_index(drop=True)

# Example
sample_title = df['title'].iloc[0]
print(f"Recommendations for: {sample_title}\n")
recommend(sample_title)


recommend(df['title'].iloc[5], n=5)


# 4. Export for the Dashboard

import json

export_records = []
for idx, row in df.iterrows():
    recs = recommend(row['title'], n=8)
    rec_titles = recs['title'].tolist() if isinstance(recs, pd.DataFrame) else []
    export_records.append({
        'id': row['show_id'],
        'title': row['title'],
        'type': row['type'],
        'director': row['director'],
        'cast': row['cast'],
        'country': row['country'],
        'release_year': int(row['release_year']),
        'rating': row['rating'],
        'duration': row['duration'],
        'genres': row['listed_in'],
        'description': row['description'],
        'recommendations': rec_titles,
    })

with open('netflix_dashboard_data.json', 'w') as f:
    json.dump(export_records, f)

print(f"Exported {len(export_records)} records to netflix_dashboard_data.json")
