import pandas as pd
import re

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# LOAD THE DATA SET #
#  - Download lyrics data set from https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/version/2
#  - Rename it to metroLyrics.csv
# Yes, this dataset has over 300k lyrics of tons of genres and artists, which is why the csv file didn't make it to this repo

#NOTE: Original dataset no longer available

dataset = pd.read_csv('metroLyrics.csv', encoding = 'UTF_8')
# print(dataset.head())
# print(dataset.describe()) #3.6M entries!


# PROCESS THE DATASET #

#filter by genre
metal_ds = dataset[dataset['genre'].str.contains('Metal')]
# print(metal_ds.describe()) #28K entries, still too much

# read all metal artists
# metal_artists = metal_ds.artist.unique()
# print ("size: " + str(len(metal_artists))) #1,155 artists!
# print("\n".join(metal_artists))

# filter by one artist
# cradle_ds = metal_ds[metal_ds['artist'].str.contains('cradle')]
# print(cradle_ds.head())
# print(cradle_ds.describe()) #204 entries, maybe too little

#filter by 10 Metal artists
metal_artists = ['after-forever', 'arch-enemy', 'blind-guardian', 'children-of-bodom',
                 'cradle-of-filth', 'dark-moor', 'dragonforce', 'edguy', 'epica']
top_ten_metal_ds = metal_ds[metal_ds['artist'].isin(metal_artists)]
# print(top_ten_metal_ds.head())
# print(top_ten_metal_ds.describe()) #1,015 entries, fair amount

# we don't care about any column but 'lyrics'
lyrics = top_ten_metal_ds.lyrics #every row is a full lyric
# print(lyrics.head())
# print(lyrics.describe())


# PROCESS THE LYRICS #

# concatenate all songs into one .txt file
with open('lyrics-ds.txt', 'w',encoding='UTF-8') as filehandle:
    for item in lyrics:
        if isinstance(item, str):
            # clean data to keep valid characters, convert to lower case
            item = re.sub("[^a-z0-9.,\n-&'?!: ]", '', item.lower())
            filehandle.write('%s\n' % item)

