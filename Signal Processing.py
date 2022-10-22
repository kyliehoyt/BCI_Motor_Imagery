from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


class TemporalFilter:
    def __init__(self, n, btype):
        self.n = n
        self.btype = btype

    def plot_frequency_response(self, b, a, fs, cutoff):
        w, h = signal.freqz(b, a, fs=fs)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        for c in cutoff:
            plt.axvline(c, color='green')  # cutoff frequency
        plt.show()
        return


class ButterFilter(TemporalFilter):
    def __init__(self, n, btype, fs, cutoff):
        super().__init__(n, btype)
        self.fs = fs
        self.cutoff = cutoff

    def apply_filter(self, raw_sig):
        b, a = signal.butter(self.n, self.cutoff, self.btype, fs=self.fs)
        return signal.filtfilt(b, a, raw_sig)

    def plot_freq_response(self):
        b, a = signal.butter(self.n, self.cutoff, self.btype, fs=self.fs)
        plt.title('Butterworth filter frequency response')
        super().plot_frequency_response(b, a, self.fs, self.cutoff)
        return


mu_filt = ButterFilter(4, 'band', 512, [8, 12])
mu_filt.plot_freq_response()

