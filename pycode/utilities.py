from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def draw(data, title=None):
    if len(data.shape) == 2:
        plt.plot(range(data.shape[0]), np.squeeze(data[:, 0:1]))
    else:
        plt.plot(range(data.shape[0]), np.squeeze(data))
    if title is not None:
        plt.title(title)
    plt.show()


def read(inputfile):
    samplerate, data = wavfile.read(inputfile)
    return samplerate, np.squeeze(data[:, 0:1])


def write(outputfile, samplerate, data):
    ret = np.empty([data.shape[0], 2], dtype=np.int16)
    for i in range(data.shape[0]):
        ret[i][0] = ret[i][1] = data[i]
    wavfile.write(outputfile, samplerate, ret)


def lowpass(inputfile, outputfile, arg):    # 低通滤波
    samplerate, data = read(inputfile)

    length = data.shape[0]
    data = np.fft.fft(data)
    l = int(arg * length)
    r = int(length - l)
    data[l:r] = np.empty([r - l], dtype=np.int16)
    data = np.real(np.fft.ifft(data))

    write(outputfile, samplerate, data)


def smooth_m(inputfile, outputfile, radius):  # 中值滤波
    samplerate, data = wavfile.read(inputfile)

    # draw(data)

    length = data.shape[0]
    out = np.empty(data.shape, dtype=np.int16)

    for i in range(0 + radius, length - radius):
        t = np.median(data[i - radius: i + radius])
        out[i][0] = out[i][1] = np.int16(t)

    draw(out)

    wavfile.write(outputfile, samplerate, out)


def smooth_a(inputfile, outputfile, radius):  # 中值滤波
    samplerate, data = wavfile.read(inputfile)

    # draw(data)

    length = data.shape[0]
    out = np.empty(data.shape, dtype=np.int16)

    for i in range(0 + radius, length - radius):
        t = np.average(data[i - radius: i + radius])
        out[i][0] = out[i][1] = np.int16(t)

    draw(out)

    wavfile.write(outputfile, samplerate, out)


def process(data, rate):
    data = np.fft.fft(data)
    length = data.shape[0]
    s = np.sort(abs(data))
    t = s[int(rate * length)]
    for i in range(length):
        if abs(data[i]) < t:
            data[i] = 0
    data = np.fft.ifft(data)
    return np.real(data)


def smooth(inputfile, outputfile, rate):
    samplerate, data = read(inputfile)

    draw(data)

    length = data.shape[0]
    seglen = 1024
    for i in range(0, length, seglen):
        data[i:i+seglen] = process(data[i:i+seglen], rate)

    draw(data)

    write(outputfile, samplerate, data)