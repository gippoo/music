# music

Python code to be posted.

## Goal:
Train a neural network to generate anime style music.

## Method:
**1. Obtain Dataset**
Downloaded 92 piano midis from various sites such as https://ichigos.com/ and http://josh.agarrado.net/music/anime/index.php</br>

Used pypianoroll (https://salu133445.github.io/pypianoroll/index.html) to convert midis into a piano roll matrix.</br>
After a bit of preprocessing, these piano rolls served as input to the network.

**2. Network Architecture**

Autoencoder with the same structure as https://github.com/HackerPoet/Composer/blob/master/train.py</br>
Less nodes in each layer as well as a MUCH smaller latent space (8 dimensions).</br>
The encoder portion of the network learns a way to represent the piano roll matrix of each song as a vector of 8 values.</br>
In theory, each of the values should represent some feature of the songs. In practice, it is difficult to determine exactly what these learned features are.</br>
The decoder portion then uses the 8 values to reconstruct the original song.


**3. Generating New Songs**

Once the network has learned to recreate the original midis with a decent amount of accuracy, the encoder portion of the network was thrown away. We can then feed random vectors of 8 values to the decoder which <i>hopefully</i> turns them into something that sounds okay.

**4. Results**

Some randomly generated songs: https://gippoo.github.io/music/</br>
Song titles are the actual song that is closest to the generated song in the latent space.
