from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sys import argv


def draw(data, title=None):
    if len(data.shape) == 2:
        plt.plot(range(data.shape[0]), np.squeeze(data[:, 0:1]))
    else:
        plt.plot(range(data.shape[0]), np.squeeze(data))
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    # samplerate, data = wavfile.read(argv[0])
    samplerate, data = wavfile.read('Record2_test.wav')
    print(data.shape)
    data = np.real(np.fft.fft(data[:,0]))
    draw(data)
