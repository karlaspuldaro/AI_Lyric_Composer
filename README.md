# Metal Composer AI (using LSTM RNN)

(OPTIONAL: Select a different genre and/or artists:
- Download lyrics data set from https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/version/2
- Rename it to metroLyrics.csv
- make changes in dataprocessing.py
- run python ./dataprocessing.py to generate cleaned up and filtered lyrics.txt )

1 - Configure python virtual environment to use tensorflow:
  virtualenv --system-site-packages -p python3 ~/tensorflow-env

2 - Activate the environment:
  source ~/tensorflow-env/bin/activate

3 - To train the model:  
  python ./composer.py train [optional_checkpoint_file]    
  (Training mode will create a checkpoint file after every epoch. The latest file can be passed to this command, the model resumes its training from there)

4 - Run the model: 
  python ./composer.py <checkpoint_file>

