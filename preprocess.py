import os
from scipy.io import wavfile
import scipy.signal as signal
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import soundfile as sf
import wave
import scipy
import eyed3
import pickle
import random
import time


# Taken from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def extract_mp3_files():
    dir_path = './../fma_small/'
    dirs = [dir_path + dir for dir in os.listdir(dir_path) if os.path.isdir(dir_path + dir)]
    mp3_files = []

    for dir in dirs:
        for f in os.listdir(dir):
            if f.endswith('.mp3'):
                try:
                    file_path = dir + '/' + f
                    # f = eyed3.load(file_path)
                    # genre = f.tag.genre.name
                    # mp3_files.append((file_path, genre))
                    mp3_files.append(file_path)
                except:
                    print(file_path + " is not a valid file")
    return mp3_files

def clip_wav(mp3_files):
    out_dir = './../wav_data/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir('./tmp'):
        os.mkdir('./tmp')
    n_files = len(mp3_files)
    for i, mp3_file in enumerate(mp3_files):
        printProgressBar(i + 1, n_files, prefix='Progress:', suffix='Complete', length=50)
        try:
            file_id = mp3_file.split('/')[-1].split('.')[0]
            sound = AudioSegment.from_mp3(mp3_file)
            wav_file = './tmp/' + file_id + '.wav'
            sound.export(wav_file, format='wav')
            duration = librosa.get_duration(path=wav_file)
            n_clips = int(duration // 2)
            audio = AudioSegment.from_wav(wav_file)
            audio = audio.split_to_mono()[0]
            for i in range(n_clips):
                new_audio = audio[2000 * i:2000 * i + 2000]
                new_audio.export(out_dir + file_id + '_' + str(i).zfill(2) + '.wav', format='wav')
            os.remove(wav_file)
        except:
            print("COULD NOT PROCESS: " + mp3_file)

def extract_wav_files():
    wav_files = []
    file_dir = './../wav_data'
    for f in os.listdir(file_dir):
        if os.path.isfile(file_dir + '/' + f):
            wav_files.append(file_dir + '/' + f)
    print(str(len(wav_files)) + " wav files found!")
    return wav_files

def build_dataset(wav_files, n_size=10000, dataset_name='dataset_wav', sample_rate=8000):
    random.seed(3)
    if n_size > len(wav_files):
        print("INVALID DATASET SIZE OF " + str(n_size))
        return -1
    n_indices = random.sample(range(0, len(wav_files)), n_size)
    dataset_dir = './../datasets'
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    data_full = []
    for i, j in enumerate(n_indices):
        wav_file = wav_files[j]
        printProgressBar(i + 1, n_size, prefix='Build Progress:', suffix='Complete', length=50)
        sig, fs = librosa.core.load(wav_file, sr=sample_rate)
        abs_spectrogram = np.abs(librosa.stft(sig))
        audio_signal = librosa.griffinlim(abs_spectrogram, length=sample_rate * 2)
        data_full.append(np.array(audio_signal))
    full_name = dataset_dir + '/' + dataset_name + '_' + str(n_size) + '.pkl'
    f_data = open(full_name, 'wb')
    pickle.dump(np.array(data_full), f_data)
    print("WROTE " + str(n_size) + " wav snippets to " + full_name)

def load_dataset(dataset_name):
    dataset_path = './../datasets/' + dataset_name
    with open(dataset_path, 'rb') as f_dataset:
        dataset = pickle.load(f_dataset)
    print(dataset.shape)
    return dataset
    

def produce_wav(wav_data, file_name, sample_rate=8000):
    if not os.path.isdir('./generated_wav/'):
        os.mkdir('./generated_wav/')
    wav_dir = './generated_wav/' + file_name
    sf.write(wav_dir, wav_data, sample_rate, 'PCM_24')



def main():
    # (1) only run once ~20 minutes runtime
    #     converts mp3 data to 2 second wav files
    # mp3_files = extract_mp3_files()
    # clip_wav(mp3_files)

    # (2) only run once ~ 20 minutes runtime for 10000 wav clips
    #     builds small, medium, large, and full sized datasets
    # wav_files = extract_wav_files()
    # build_dataset(wav_files, n_size=100)
    # build_dataset(wav_files, n_size=10000)
    # build_dataset(wav_files, n_size=25000)
    # build_dataset(wav_files, n_size=100000)
    # build_dataset(wav_files, n_size=len(wav_files))

    # (3) no need to run
    #     example of how to read dataset
    # wav_dataset = load_dataset('dataset_wav_10000.pkl')
    # produce_wav(wav_dataset[1], 'test.wav')
    return 0

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("TIME ELAPSED: " + "{:.2f}".format(t_end - t_start) + " SECONDS")
