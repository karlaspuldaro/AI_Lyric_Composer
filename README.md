# Heavy Metal Composer AI
Artificial Intelligence Heavy Metal Songwriter built under a Long Short-term Memory ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) Recurrent Neural Network ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)) architecture.
Song lyrics originally obtained from the [Kaggle dataset](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/version/2)

*Note: Original full dataset no longer available.
`lyrics-ds.txt` is a filtered dataset which includes over 1k songs from a selection of 10 artists (described in `dataprocessing.py`).

---
## Requirements
- Conda
- JupyterLab
- Elyra
- Numpy
- Tensorflow

---
## Installation
1. Configure conda virtual environment for python:

    This will help managing dependencies and isolate our project

    `conda create -n myenv python=3.7`

2. Activate the environment:

    `conda activate myenv`

3. Install dependencies:
- numpy:

    `conda install -c anaconda numpy`

- tensorflow:

   `conda install -c conda-forge tensorflow`

4. Install JupyterLab and Elyra:

    `conda install -c conda-forge jupyterlab`

    `conda install -c conda-forge elyra`

5. Build JupyterLab

    `jupyter lab build`

6. Verify Installation

    `jupyter serverextension list && jupyter labextension list`

---
## Build and run JupyterLab

  `jupyter lab build`

  In the root of your clone of this github project, run
  `jupyter lab`

---
## Open JupyterLab and run the model

  Once JupyterLab is launched in your web browser, all files from this repository will be loaded and are accessible from the File Browser on the left pane of JupyterLab's main page.

  Select `composer-notebook.ipynb` file and open it.
  Run the notebook (Run tab --> Run All)

---
## Model training
  This repository includes 2 trained models, saved as checkpoints in hdf5 files.

  To re-train the model, from a terminal, run

  `composer.py train <optional_checkpoint_file>`

  Training mode will create a checkpoint file after every epoch. The latest file can be passed to this command, the model resumes its training from there.
