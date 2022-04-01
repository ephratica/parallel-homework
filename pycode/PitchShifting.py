import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import TimeScaling


def pitchShift(data, rate):
    data = TimeScaling.timeScale(data, rate)
    lenth = data.shape[0]
    newdata = np.empty([int(lenth / rate) + 1, 2], dtype=np.int16)
    for i in range(int(lenth / rate)):
        pos = int(i * rate)
        newdata[i] = data[pos]
    return newdata


def draw(data):
    plt.plot(range(data.shape[0]), np.squeeze(data[:, 0:1]))
    plt.show()


if __name__ == '__main__':
    samplerate, data = wavfile.read('Record2.wav')

    # print(data.shape)
    # print(data)
    draw(data)
    data = pitchShift(data, 1.25)
    draw(data)
    # print(data.shape)
    # print(data)

    wavfile.write('Record2_out2.wav', samplerate, data)
