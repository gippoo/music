# music

## Goal:
Train a neural network to generate music in the likeness of some of my rpg games.

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
221 songs total</br>

Used pypianoroll (https://salu133445.github.io/pypianoroll/index.html) to convert midis into a piano roll matrix.</br>
After a bit of preprocessing, these piano rolls served as input to the network.

**2. Network Architecture**

Autoencoder with the same structure as https://github.com/HackerPoet/Composer/blob/master/train.py</br>
Less nodes in each layer as well as a smaller latent space.


**3. Generating New Songs**

Once the network has learned to recreate the original midis, the encoder portion of the network was thrown away. The decoder can then be used to turn random noise into something that <i>hopefully</i> sounds okay.

**4. Results**
Some randomly generated songs: https://soundcloud.com/gippooo/sets/neural-network-music/s-erPJb
Song titles are the actual song that is closest to the generated song in the latent space.

