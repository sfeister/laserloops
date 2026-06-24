# Creating and analyzing data batches, all local (no HPC)

This demo involves creating a few batches of data, training an AI model on that dataset, and then moving weights from that model back to a folder and making predictions.

## Python environment for this demo
Here's how I set up my Python environment:

pip install --upgrade numpy scipy jupyterlab pandas matplotlib ipywidgets h5py scikit-learn ipykernel joblib tqdm

## Running the Demo
From your terminal, here are the steps to run the demo.

1. `python make_batches.py` Pretends to be a data acquisition in an experiment lab. The laboratory (imaginary) has two instruments: a proton spectrometer and an electron spectrometer. Both emit 1D trace data of 100 elements, for every laser shot. This script makes three batches of 2000 shots each, and stores them as HDF5 files in `./outputs/batches/` (The folder will be created if it doesn't already exist.) You can repeat running this as many times as you wish and it will just keep making more and more batches of data. Each batch of data is given an incremented batch number, like `./outputs/batches/batch_001`, `./outputs/batches/batch_002`, ...
2. `python train.py` Trains a simple ML model based on all the ELECTRON/PROTON data found in `./outputs/batches`. The goal of this model is to predict a 100-point PROTON spectrum based on a 100-point ELECTRON spectrum. Saves the model weights to `./outputs/models/model_001`. You can run this again any time you'd like, and it will re-train a new model based on all data in `./outputs/batches`. The new model will be given a new, incremented model number, like `./outputs/models/model_002`, `./outputs/models/model_003`, etc.
3. `python predict_plot.py` Takes a new data acquisition (a single shot of data) and (using the latest model in `./outputs/models/` predicts what the PROTON spectrum should look like based on the ELECTRON spectrum. We then make a graph of the prediction vs. the reality. You can run this as many times as you'd like. 

You can run any of these elements over and over, and see the predictions improving as you increase the training data (if you'd like.)
