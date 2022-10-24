from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import math
import seaborn as sns

LargeNeighbors = [[2, 4],
                  [5],
                  [0, 6],
                  [4],
                  [0, 2, 5, 14],
                  [1, 4, 6, 14],
                  [2, 5, 7, 16],
                  [6],
                  [9, 19],
                  [8, 10, 20],
                  [9, 11, 21],
                  [10, 22],
                  [13, 19],
                  [3, 12, 14, 23],
                  [4, 13, 14, 24],
                  [1, 14, 16, 25],
                  [6, 15, 17, 26],
                  [7, 16, 18, 27],
                  [17, 22],
                  [8, 20],
                  [9, 19, 21],
                  [10, 20, 22],
                  [11, 21],
                  [13, 24, 29],
                  [14, 23, 25, 29],
                  [15, 24, 26, 30],
                  [16, 25, 27, 31],
                  [17, 26, 31],
                  [24, 26, 29, 31],
                  [23, 24, 31],
                  [25],
                  [26, 27, 29]]


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
        return signal.filtfilt(b, a, raw_sig, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

    def plot_freq_response(self):
        b, a = signal.butter(self.n, self.cutoff, self.btype, fs=self.fs)
        plt.title('Butterworth filter frequency response')
        super().plot_frequency_response(b, a, self.fs, self.cutoff)
        return


def laplacian_filter(neighborhood, h, s):
    lap_filt = np.identity(len(neighborhood))
    for r in range(len(lap_filt)):
        sum_dij = sum(distance(h, r, neighborhood(r)))
        lap_filt[r, neighborhood[r]] = [-1 * dij / sum_dij for dij in distance(h, r, neighborhood[r])]
    return np.matmul(s, lap_filt)


def distance(h, c1, c2):
    return [((h(c).X - h(c1).X) ** 2 + (h(c).Y - h(c1).Y) ** 2 + (h(c).Z - h(c1).Z) ** 2) ** 0.5 for c in c2]


def car_filter(s):
    dim = 32
    car_filt = -1 / dim * np.ones((dim, dim)) + np.identity(dim)
    if np.ndim(s) == 2:
        return np.matmul(s, car_filt)
    elif np.ndim(s) == 3:
        for tr in range(np.shape(s)[2]):
            s[:, :, tr] = np.matmul(s[:, :, tr], car_filt)
        return s


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def runs2trials_split(ss, hs, dur):
    win = math.floor(dur * hs[0]['SampleRate'])
    s_L = np.zeros((win, 32, len(ss)*10))
    s_R = np.zeros((win, 32, len(ss)*10))
    lt = 0
    rt = 0
    for s, h in zip(ss, hs):
        trigs = h['EVENT']['TYP']
        L_pos = [h['EVENT']['POS'][i] for i, v in enumerate(trigs) if v == 7692 or v == 7693]
        R_pos = [h['EVENT']['POS'][i] for i, v in enumerate(trigs) if v == 7702 or v == 7703]
        for l, r in zip(L_pos, R_pos):
            s_L[:, :, lt] = s[l - win:l, :]
            lt = lt + 1
            s_R[:, :, rt] = s[r - win:r, :]
            rt = rt + 1
    return s_L, s_R

def runs2trials(ss, hs, dur):
    win = math.floor(dur * hs[0]['SampleRate'])
    trs = np.zeros((win, 32, len(ss)*20))
    t = 0
    for s, h in zip(ss, hs):
        trigs = h['EVENT']['TYP']
        pos = [h['EVENT']['POS'][i] for i, v in enumerate(trigs) if v in [7692, 7693, 7702, 7703]]
        for p in pos:
            trs[:, :, t] = s[p - win:p, :]
            t = t + 1
    return trs


def run_psd(s, h, fmax):
    maxfbin = int(fmax/2)
    s_psd = np.zeros((maxfbin, np.shape(s)[1], np.shape(s)[2]))
    for tr in range(np.shape(s_psd)[2]):
        for ch in range(np.shape(s_psd)[1]):
            f, s_psd[:, ch, tr] = np.array(signal.periodogram(s[:, ch, tr], fs=h['SampleRate'], scaling='density'))[:, :maxfbin]
    s_psd_avg = np.mean(s_psd, 2)
    s_psd_avg = np.swapaxes(s_psd_avg/np.max(s_psd_avg), 0, 1)
    ylabels = [str.strip() for str in h['Label'][:np.shape(s)[1]]]
    sns.heatmap(s_psd_avg, xticklabels=f, yticklabels=ylabels)
    return


h1 = loadmat('h1.mat')['h1']
h2 = loadmat('h2.mat')['h2']
h3 = loadmat('h3.mat')['h3']
h4 = loadmat('h4.mat')['h4']
s1 = loadmat('s1.mat')['s1']
s2 = loadmat('s2.mat')['s2']
s3 = loadmat('s3.mat')['s3']
s4 = loadmat('s4.mat')['s4']
chan_info = loadmat('chaninfo.mat')['selectedChannelstruct']['selectedChannels']

fs = h1['SampleRate']
broad = [4, 30]
broad_filt = ButterFilter(2, 'band', fs, broad)
runs = [s1, s2, s3, s4]
heads = [h1, h2, h3, h4]

plt.subplot(1, 4, 1)
i = 1
for s, h in zip(runs, heads):
    s = broad_filt.apply_filter(s)
    s_tr = runs2trials([s], [h], 0.5)
    s_tr_filt = car_filter(s_tr)
    plt.subplot(1, 4, i)
    i = i+1
    run_psd(s_tr_filt, h, 32)

plt.show()

