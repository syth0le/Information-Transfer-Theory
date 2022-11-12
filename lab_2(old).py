import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy import signal

SAMPLE_RATE = 700  # Hertz
DURATION = 5  # Seconds


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


def create_normalized_tone():
    _, nice_tone = generate_sine_wave(20, SAMPLE_RATE, DURATION)
    _, noise_tone = generate_sine_wave(40, SAMPLE_RATE, DURATION)
    noise_tone *= 0.3
    mixed_tone = nice_tone + noise_tone
    normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)
    return normalized_tone


normalized_tone = create_normalized_tone()

N = SAMPLE_RATE * DURATION
yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

fast_yf = rfft(normalized_tone)
fast_xf = rfftfreq(N, 1 / SAMPLE_RATE)

window = signal.windows.hamming(51)
A = fft(window, N) / (len(window) / 2)
freq = np.linspace(min(xf), max(xf), len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

def draw_graphic():
    plt.axis([min(xf), max(xf), 0, 500])
    plt.plot(xf, np.abs(yf), color='red', linewidth= 4)
    plt.plot(fast_xf, np.abs(fast_yf), color='blue')
    plt.plot(freq, [x + 100 for x in response], color='green')
    plt.xlabel('Частота')
    plt.ylabel('Частота')
    plt.show()

