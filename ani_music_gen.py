import numpy as np
import os
from pypianoroll import Multitrack, Track, parse
import matplotlib.pyplot as plt

directory = 'anime'
beat_resolution = 4
pad_length = 576
seq_len = 96
latent_space = 8
batch_size = 23
epochs = 750


# Parsing the midi files
def parse_midis():
    midis = []

    for midi in os.listdir('./'+directory):
        print('Parsing '+midi)
        a_midi = parse('./'+directory+'/' + midi, beat_resolution=beat_resolution)
        midis.append(a_midi)

    print('SONGS LOADED')

    return midis


# For shorter songs, we pad the piano rolls with 0's until the desired length
def pad_piano_rolls(midis):
    ix = 0
    
    for midi in midis:
        proll_len = midi.get_merged_pianoroll(mode='sum').shape[0]
        
        if proll_len < pad_length:
            print("Padding " + os.listdir('./'+directory)[ix] + '...')
            print("Original Length: " + str(proll_len))
            midi.pad(pad_length - proll_len)
            
        ix += 1

    return midis


# Function to turn notes into 1's and 0's to represent if it is played or not.
def onezero(note):
    if note > 0:
        return 1
    else:
        return 0


binarize = np.vectorize(onezero)


def binarize_prolls(midis):
    piano_rolls = []

    print('LOADING PIANO ROLLS...')

    for midi in midis:
        piano_rolls.append(binarize(midi.get_merged_pianoroll(mode='sum')))

    print('PIANO ROLLS LOADED')

    return piano_rolls


# Break the piano rolls into sequences of time steps
def preprocess_prolls(piano_rolls):
    num_parts = int(np.floor(pad_length/seq_len))

    final_rolls = []

    print('PREPROCESSING INPUT DATA...')

    roll_count = 1
    for roll in piano_rolls:
        froll = []
        
        for i in range(num_parts):
            froll.append(roll[i*seq_len:(i+1)*seq_len])
        
        roll_count += 1
        final_rolls.append(froll)

    final_rolls = np.asarray(final_rolls)
    print('\nCOMPELETE \nINPUT SHAPE: ' + str(final_rolls.shape))

    return final_rolls, num_parts


# Create our training data
x = parse_midis()
x = pad_piano_rolls(x)
x = binarize_prolls(x)
x, parts = preprocess_prolls(x)


# Build and train the model
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Flatten, BatchNormalization
from keras.layers import Dropout
from keras.layers import TimeDistributed, Activation
from keras import backend as K

input_song = Input(shape=x.shape[1:])
q = Reshape((parts, -1))(input_song)
q = TimeDistributed(Dense(1200, activation='relu'))(q)
q = TimeDistributed(Dense(120, activation='relu'))(q)
q = Flatten()(q)
q = Dense(300, activation='relu')(q)
q = Dense(latent_space)(q)
q = BatchNormalization(momentum=0.9, name='encoded')(q)

q = Dense(300, name='decoder')(q)
q = BatchNormalization(momentum=0.9)(q)
q = Activation('relu')(q)
q = Dropout(0.1)(q)

q = Dense(parts * 120)(q)
q = Reshape((parts, 120))(q)
q = BatchNormalization(momentum=0.9)(q)
q = Activation('relu')(q)
q = Dropout(0.1)(q)

q = TimeDistributed(Dense(1200))(q)
q = TimeDistributed(BatchNormalization(momentum=0.9))(q)
q = Activation('relu')(q)
q = Dropout(0.1)(q)

q = TimeDistributed(Dense(seq_len*128, activation='sigmoid'))(q)
q = Reshape((parts, seq_len, 128))(q)

autoencoder = Model(input_song, q)

autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

decoder = K.function([autoencoder.get_layer('decoder').input], [autoencoder.layers[-1].output])
encoder = K.function([autoencoder.layers[0].input], [autoencoder.get_layer('encoded').output])

autoencoder.fit(x, x, epochs=epochs, shuffle=True, batch_size=batch_size)


# Function to find the most similar song
songvecs = encoder([x])
def closest_song(noise_vec):
    ix = 0
    smallest_dist = 100000
    smallest_ix = 0
    
    for i in songvecs[0]:
        d = np.sqrt(np.sum(np.square(i - noise_vec.reshape(latent_space))))
        if d < smallest_dist:
            smallest_dist = d
            smallest_ix = ix
        ix += 1
        
    print('\nCLOSEST SONG: '+os.listdir('./'+directory)[smallest_ix])
    print(songvecs[0][smallest_ix])
    print('DISTANCE: '+str(smallest_dist))


# Function for how sure the network has to be to play a note
def threshold(note):
    if note > np.random.uniform(0.4, 0.6):
        return 1
    else:
        return 0


playnotes = np.vectorize(threshold)


# Generating a random song
def gen_song(tempo, title, plot_proll=False):
    print('\n GENERATING SONG...')
    noise = np.random.normal(0, 1, (1, latent_space))
    print(noise)
    
    new_song = playnotes(np.array(decoder([noise])).reshape(parts*seq_len, 128))*100
    track = Track(new_song, name='test')
    
    if plot_proll:
        track.plot()
        plt.show()
        
    multitrack = Multitrack(tracks=[track], beat_resolution=beat_resolution, tempo=tempo)
    multitrack.write(title+'.mid')
    closest_song(noise)


gen_song(90, 'test')
