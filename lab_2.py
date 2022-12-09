import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy import signal


class FFT:

    def __init__(self, sample_rate, duration):
        self.sample_rate = sample_rate
        self.duration = duration
        self.N = self.sample_rate * self.duration

    def generate_sine_wave(self, freq):
        x = np.linspace(0, self.duration, self.sample_rate * self.duration, endpoint=False)
        frequencies = x * freq
        y = np.sin((2 * np.pi) * frequencies)
        return x, y

    def create_normalized_tone(self):
        _, nice_tone = self.generate_sine_wave(20)
        _, noise_tone = self.generate_sine_wave(40)
        noise_tone *= 0.3
        mixed_tone = nice_tone + noise_tone
        return np.int16((mixed_tone / mixed_tone.max()) * 32767)

    @staticmethod
    def _draw_graphic(xf, yf, fast_xf, fast_yf, freq, response):
        plt.axis([-100, 0, -150, 0])
        plt.plot(fast_xf, 10 * np.log10(np.abs(fast_yf) / np.max(np.abs(fast_yf))), color='blue')
        plt.plot(xf, 10 * np.log10(np.abs(yf) / np.max(np.abs(yf))), color='red', linewidth=1)
        plt.xlabel('Частота')
        plt.ylabel('Децибелы')
        plt.show()
        plt.plot(freq, [x + 100 for x in response], color='green')
        plt.xlabel('Частота')
        plt.ylabel('Децибелы')
        plt.show()

    def count_fft(self):
        normalized_tone = self.create_normalized_tone()
        window = signal.windows.blackman(3500)
        res = [elem[0] * elem[1] for elem in list(zip(normalized_tone, window))]
        yf = fft(res)
        yf_before = fft(normalized_tone)
        xf = fftfreq(self.N, 1 / self.sample_rate)

        fast_yf = rfft(res)
        fast_xf = rfftfreq(self.N, 1 / self.sample_rate)

        A = fft(window, self.N) / (len(window) / 2)
        freq = np.linspace(min(xf), max(xf), len(A))
        response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

        self._draw_graphic(xf, yf, xf, yf_before, freq, response)


if __name__ == '__main__':
    counter = FFT(sample_rate=700, duration=5)
    counter.count_fft()
