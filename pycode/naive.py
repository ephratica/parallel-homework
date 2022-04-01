import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def process(data, rate):
    l = len(data)
    mid = int(l / 2 + 0.5)
    t = 1 / rate
    ret = np.empty(np.shape(data), dtype=complex)
    for i in range(mid):
        ret[i] = data[int(i * t)]
    for i in range(mid, l):
        ret[i] = data[l - 1 - int((l - i) * t)]
    return ret


#
# def process2(data, rate):
#     l = len(data)
#     mid = int(l / 2 + 0.5)
#     t = int(mid * rate)
#     ret = np.empty(np.shape(data), dtype=complex)
#     for i in range(t, mid):
#         ret[i] = data[i - t]
#     for i in range(mid, l - t):
#         ret[i] = data[i + t]
#     return ret


def work(data):
    outdata = data
    data = np.squeeze(data[:, 0:1])

    data = np.fft.fft(data)

    # out = process2(data, 0.05)
    out = process(data, 2)

    # plt.plot(range(len(data)), list(data))
    # plt.plot(range(len(data)), list(out))
    # plt.show()

    out = np.fft.ifft(out)
    out = np.real(out)

    lenth = out.shape[0]
    # print(lenth)
    for i in range(lenth):
        outdata[i][0] = outdata[i][1] = out[i]
    return outdata


if __name__ == '__main__':
    samplerate, data = wavfile.read('Record2.wav')
    print(samplerate, data)

    plt.plot(list(range(len(data))), [t[0] for t in data])
    plt.show()

    seglen = 512
    for i in range(0, data.shape[0], seglen):
        data[i:i + seglen] = work(data[i:i + seglen])

    plt.plot(list(range(len(data))), [t[0] for t in data])
    plt.show()

    wavfile.write('Record2_out_naive.wav', samplerate, data)