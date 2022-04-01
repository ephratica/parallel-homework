import math
import numpy as np

import TimeScaling
import PitchShifting
import utilities
from scipy.io import wavfile
import matplotlib.pyplot as plt
from utilities import draw

if __name__ == '__main__':
    X = np.arange(0.01, math.pi - 0.01, 0.01)
    M = 5
    f = lambda x: abs(1 * math.sin(M * x / 2) / math.sin(x / 2) / M)
    Y = [f(x) for x in X]
    plt.plot(X, Y)
    plt.show()

# if __name__ == '__main__':
#     # utilities.smooth('Record2_out.wav', 'Record2_out_s.wav', 0.9)
#     # utilities.smooth_m('Record2_out.wav', 'Record2_out_m.wav', 2)
#     utilities.smooth_a('Record2_out.wav', 'Record2_out_a.wav', 2)
