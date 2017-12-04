import os
import csv
import umap
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
from os.path import join
from tqdm import tqdm
np.random.seed(8)

data_root = 'drumData'
n_fft = 1024
hop_length = n_fft/4
use_logamp = False # boost the brightness of quiet sounds
reduce_rows = 10 # how many frequency bands to average into one
reduce_cols = 1 # how many time steps to average into one
crop_rows = 32 # limit how many frequency bands to use
crop_cols = 32 # limit how many time steps to use
limit = None # set this to 100 to only process 100 samples

drumNames = ["kick", "tom", "snare", "clap", "hi.hat", "ride", "crash"]
drumFingerPrints = {}
drumSamples = {}
for d in drumNames:
    drumSamples[d] = np.load(join(data_root, d+'_samples.npy'))
for d in drumNames:
    print (d, drumSamples[d].shape)

import tensorflow as tf
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet.h512_bo16 import FastGenerationConfig

def custom_encode(wav_data, checkpoint_path, drum_name, sample_length=64000):
    batch_size = 1
    samples_wavenet = []
    num_samples = wav_data.shape[0]
    # Load up the model for encoding and find the encoding of "wav_data"
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        hop_length = Config().ae_hop_length
        net = testload_nsynth(batch_size=batch_size, sample_length=11776)  # hardcore to 11776 for samples of length 12000
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        for i in tqdm(range(num_samples)):
            sample = wav_data[i]
            sample = np.expand_dims(sample,0)
            sample, sample_length = utils.trim_for_encoding(sample, sample_length,hop_length)
            encodings = sess.run(net["encoding"], feed_dict={net["X"]: sample})
            encodings = encodings.reshape(-1,16)
            samples_wavenet.append(encodings)
        samples_wavenet = np.asarray(samples_wavenet)
        print (drum_name, samples_wavenet.shape)
        file_path = './drumData/' + drum_name + '_wavenet.npy'
        np.save(file_path, samples_wavenet)

def testload_nsynth(batch_size=1, sample_length=64000):
    """Load the NSynth autoencoder network.
    Args:
    batch_size: Batch size number of observations to process. [1]
    sample_length: Number of samples in the input audio. [64000]
    Returns:
    graph: The network as a dict with input placeholder in {"X"}
    """
    config = Config()
    with tf.device("/gpu:0"):
        x = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
        graph = config.build({"wav": x}, is_training=False)
        graph.update({"X": x})
    return graph

drum_name = raw_input("Enter drum name type to encode:")
custom_encode(drumSamples[drum_name], './wavenet-ckpt/model.ckpt-200000',drum_name, 12000)
