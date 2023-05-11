# MusicSynthesis

by [Bryton Foster](https://github.com/b-foster-yale), [Neal Ma](https://github.com/nmadev), and [Michael Ying](https://github.com/mying2002)

## Generated audio files

Sample generated audio files (``.wav`` format) with different generating procedures can be found [here](https://github.com/nmadev/MusicSynthesis/tree/main/generated_wav). [``FILE PARAMETERS``](https://github.com/nmadev/MusicSynthesis/blob/main/generated_wav/FILE_PARAMETERS) contains the parameters used to create each file. Generated files and their corresponding original files are found [here](https://github.com/nmadev/MusicSynthesis/tree/main/generated_wav/merged_wav).

## Preprocess Utility

To use the ``preprocess.py`` there is a specific file stucture that you must use the structure:
````{verbatim}
  project/
  |--CPSC-452-Final-Project/
  |  |--preprocess.py
  |--fma-small/
  |  |--000/
  |  |--001/
  |  |--...
````
Here we have an overall encapsulating directory, ``project``, containing a copy of this repository as well as the extracted ``fma-small`` dataset. This seperation is necessary to ensure that pushing we do not push enormous amounts of data to this repository. 

Uncommenting ``(1)`` in ``preprocess.py`` will extract all the .mp3 file names from ``fma-small`` and produce 2 second .wav files extracted from the .mp3 files. These files are all stored in a directory ``wav_data`` stored directly inside ``project``. This operation should take about 20 minutes on a cpu and progress can be tracked with the progress bar. After this operation, the file tree should look like:
````{verbatim}
  project/
  |--CPSC-452-Final-Project/
  |  |--preprocess.py
  |--fma-small/
  |  |--000/
  |  |--001/
  |  |--...
  |--wav_data/
  |  |--000001.wav
  |  |--000002.wav
  |  |--...
````
Uncommenting ``(2)`` will build a dataset of a given size from the .wav files. This dataset is saved to a folder ``datasets`` directly under ``project``. After saving a dataset of 10000 2-second clips (which takes about 20 minutes), the file tree would look like this:
````{verbatim}
  project/
  |--CPSC-452-Final-Project/
  |  |--preprocess.py
  |--fma-small/
  |  |--000/
  |  |--001/
  |  |--...
  |--wav_data/
  |  |--000001.wav
  |  |--000002.wav
  |  |--...
  |--datasets/
  |  |--dataset_wav_10000.pkl
````
Note that ``dataset_wav_10000.pkl`` is about 600 MB and stores a 10000x16000 numpy array. Each row is a song. This dataset takes less than a second to load so the most intensive part is the preprocessing.

``(3)`` is just an example showing how to load the actual dataset once it has been produced and an example of converting a song from data back to .wav form. If you import ``preprocess.py`` all of these utility functions can be directly used. The most useful of which is probably  ``load_dataset()`` which takes the name of the dataset and returns a numpy array representing the dataset. 
