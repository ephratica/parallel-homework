import matplotlib.pyplot as plt
import numpy as np
import cmath


def STFT(data, seglen):  # ret: amplitude,phase
    ra = []
    rp = []
    for i in range(0, len(data), seglen):
        tmp = data[i: i + seglen]
        tmp = np.fft.fft(tmp)
        ra.append([abs(c) for c in tmp])
        rp.append([cmath.phase(c) for c in tmp])
    return ra, rp


def ISTFT(A, P):
    data = []
    for a, p in zip(A, P):
        data += list(np.fft.ifft([complex(x * np.cos(y), x * np.sin(y)) for x, y in zip(a, p)]))
    return data


def STFT2(data, seglen):  # ret: amplitude,phase
    ra = []
    rp = []
    for i in range(0, len(data), seglen):
        tmp = data[i: i + seglen]
        tmp = np.fft.fft(tmp)
        # ra.append([abs(c) for c in tmp])
        # rp.append([cmath.phase(c) for c in tmp])
        ra.append([c.real for c in tmp])
        rp.append([c.imag for c in tmp])
    return ra, rp


def ISTFT2(A, P):
    data = []
    for a, p in zip(A, P):
        # data += list(np.fft.ifft([complex(x * np.cos(y), x * np.sin(y)) for x, y in zip(a, p)]))
        data += list(np.fft.ifft([complex(x, y) for x, y in zip(a, p)]))
    return data


def timeScale(Data, rate):
    lenth = np.shape(Data)[0]
    data = np.squeeze(Data[:, 0:1])
    seglen = 1024

    # print(data)
    A, P = STFT(data, seglen)

    P_ = P
    for i in range(1, len(P)):
        for j in range(0, len(P_[i])):
            P_[i][j] = P_[i - 1][j] + rate * (P[i][j] - P[i - 1][j] + j)

    A2 = []
    P2 = []
    for a, p in zip(A, P_):
        len2 = int(len(a) * rate)
        A2.append([a[int(i / rate)] for i in range(len2)])
        P2.append([p[int(i / rate)] for i in range(len2)])

    data = ISTFT(A2, P2)
    # print(np.array(data))
    return np.array([[np.int16(x.real + 0.5), np.int16(x.real + 0.5)] for x in data])
