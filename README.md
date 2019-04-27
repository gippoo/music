# music

Python code to be posted.

## Goal:
Train a neural network to generate music in the likeness of some of my favorite rpg games.

## Method:
**1. Obtain Dataset**
Downloaded a bunch of midi files from the following games:</br>
-Final Fantasy 8</br>
-Final Fantasy 9</br>
-Final Fantasy 10</br>
-Chrono Trigger</br>
-Chrono Cross</br>
-Breath of Fire 3</br>
-Ocarina of Time</br>
187 songs total</br>

Used pypianoroll (https://salu133445.github.io/pypianoroll/index.html) to convert midis into a piano roll matrix.</br>
After a bit of preprocessing, these piano rolls served as input to the network.

**2. Network Architecture**

Autoencoder with the same structure as https://github.com/HackerPoet/Composer/blob/master/train.py</br>
Less nodes in each layer as well as a MUCH smaller latent space (10 dimensions).</br>
The encoder portion of the network learns a way to represent the piano roll matrix of each song as a vector of 10 values.</br>
In theory, each of the values should represent some feature of the songs. In practice, it may be difficult to determine exactly what these learned features are.</br>
The decoder portion then uses the 10 values to reconstruct the original song.


**3. Generating New Songs**

Once the network has learned to recreate the original midis with a decent amount of accuracy, the encoder portion of the network was thrown away. We can then feed random vectors of 10 values to the decoder which <i>hopefully</i> turns them into something that sounds okay.

**4. Results**

Some randomly generated songs: https://gippoo.github.io/music/</br>
Song titles are the actual song that is closest to the generated song in the latent space.

**5. Issues**

Several of the original songs were played with multiple instruments and are very complex. When getting the piano roll matricies, they were very "noisy" and the network learned a lot of this noise instead of the main chords and the melody. This can be heard in the generated songs where they seem to be a bit too cluttered with notes.</br>
Additonally, When preprocessing the piano rolls, I chose to binarize the notes so note volume was not preserved. Instead, all notes are played at the same volume.
