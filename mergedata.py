import pandas as pd

files = {
    'Netflix': 'netflix_titles.csv',
    'Amazon Prime': 'amazon_prime_titles.csv',
    'Disney+': 'disney_plus_titles.csv',
}

frames = []
for platform, path in files.items():
    df = pd.read_csv(path)
    df['platform'] = platform

    prefix = {'Netflix': 'nf', 'Amazon Prime': 'az', 'Disney+': 'dp'}[platform]
    df['show_id'] = prefix + '_' + df['show_id'].astype(str)
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)

# Basic cleanup shared across the combined set
merged['director'] = merged['director'].fillna('Unknown')
merged['cast'] = merged['cast'].fillna('Unknown')
merged['country'] = merged['country'].fillna('Unknown')
merged['rating'] = merged['rating'].fillna('Not Rated')
merged['listed_in'] = merged['listed_in'].fillna('Unknown')
merged['description'] = merged['description'].fillna('')

# Drop exact duplicate titles+release_year that appear on more than one platform
# but keep track of which platforms carry them (many titles are on multiple services)
merged = merged.drop_duplicates(subset=['title', 'release_year', 'platform'])

merged.to_csv('merged_titles.csv', index=False)
print(merged.shape)
print(merged['platform'].value_counts())
print(merged.isna().sum())