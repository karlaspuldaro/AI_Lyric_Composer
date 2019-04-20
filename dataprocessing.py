import pandas as pd
import re

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# LOAD THE DATA SET #
dataset = pd.read_csv('metroLyrics.csv', encoding = 'UTF_8')
# print(dataset.head())
# print(dataset.describe()) #3.6M entries, way too much


# PROCESS THE DATASET #

#filter by gender
metal_ds = dataset[dataset['genre'].str.contains('Metal')]
# print(metal_ds.describe()) #28K entries, still too much

# read all metal artists
# metal_artists = metal_ds.artist.unique()
# print ("size: " + str(len(metal_artists))) #1,155 artists!
# print("\n".join(metal_artists))

# filter by an artist
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
with open('lyrics.txt', 'w',encoding='UTF-8') as filehandle:
    for item in lyrics:
        if isinstance(item, str):
            # clean data to keep valid characters, convert to lower case
            item = re.sub("[^a-z0-9.,\n-&'?!: ]", '', item.lower())
            filehandle.write('%s\n' % item)

