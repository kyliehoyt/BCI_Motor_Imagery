import scipy.stats
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import math
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


class SpatialFilter:
    def __init__(self, name):
        self.type = name

    def filter_raw_sig(self, s, filt):
        return np.matmul(s, filt)


class LaplacianFilter(SpatialFilter):
    def __init__(self, name, neighborhood, h):
        super().__init__(name)
        self.neighborhood = neighborhood
        self.filter = np.identity(len(self.neighborhood))
        for r in range(len(self.filter)):
            sum_dij = sum(self.distance(h, r, self.neighborhood(r)))
            self.filter[r, self.neighborhood[r]] = [-1 * dij / sum_dij for dij in
                                                    self.distance(h, r, self.neighborhood[r])]

    def distance(self, h, c1, c2):
        return [((h(c).X - h(c1).X) ** 2 + (h(c).Y - h(c1).Y) ** 2 + (h(c).Z - h(c1).Z) ** 2) ** 0.5 for c in c2]

    def apply_filter(self, s):
        super().filter_raw_sig(s, self.filter)

    # def display filter


class CARFilter(SpatialFilter):
    def __init__(self, nchan):
        self.dim = nchan
        self.filter = -1 / self.dim * np.ones((self.dim, self.dim)) + np.identity(self.dim)

    def apply_filter(self, s):
        if np.ndim(s) == 2:  # raw data (samples, channels)
            super().filter_raw_sig(s, self.filter)
        elif np.ndim(s) == 3:  # trial data (win, channels, trials)
            for tr in range(np.shape(s)[2]):
                s[:, :, tr] = np.matmul(s[:, :, tr], self.filter)
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
    s_L = np.zeros((win, 13, len(ss) * 10))
    s_R = np.zeros((win, 13, len(ss) * 10))
    lt = 0
    rt = 0
    for s, h in zip(ss, hs):
        trigs = h['EVENT']['TYP']
        l_pos = [h['EVENT']['POS'][i] for i, v in enumerate(trigs) if v == 7692 or v == 7693]
        r_pos = [h['EVENT']['POS'][i] for i, v in enumerate(trigs) if v == 7702 or v == 7703]
        for l, r in zip(l_pos, r_pos):
            s_L[:, :, lt] = s[l - win:l, :]
            lt = lt + 1
            s_R[:, :, rt] = s[r - win:r, :]
            rt = rt + 1
    return s_L, s_R


def runs2trials(ss, hs, dur):
    win = math.floor(dur * hs[0]['SampleRate'])
    trs = np.zeros((win, 13, len(ss) * 20))  # trs shape is (samples, channels, trials)
    t = 0
    for s, h in zip(ss, hs):
        triggers = h['EVENT']['TYP']
        pos = [h['EVENT']['POS'][i] for i, v in enumerate(triggers) if v in [7692, 7693, 7702, 7703]]
        for p in pos:
            trs[:, :, t] = s[p - win:p, :]
            t = t + 1
    return trs


def run_psd_fisher(s, h, flim, ylab):
    fmin_bin = int(flim[0] / 2)
    fmax_bin = int(flim[1] / 2) + 1
    c = np.shape(s)[0]
    n_chan = np.shape(s)[2]
    n_tr = np.shape(s)[3]
    s_psd = np.zeros((fmax_bin-fmin_bin, n_chan, n_tr, c))
    for j in range(c):
        for tr in range(n_tr):
            for ch in range(n_chan):
                f, s_psd[:, ch, tr, j] = np.array(
                    signal.periodogram(s[j, :, ch, tr], fs=h['SampleRate'], scaling='density'))[:, fmin_bin:fmax_bin]
    # s_psd shape is (freq, channels, trials, classes)
    mean_intra = np.mean(s_psd, 2)
    var_intra = np.var(s_psd, 2)
    mean_inter = np.array([np.mean(mean_intra, 2) for t in range(c)])
    mean_inter = np.moveaxis(mean_inter, 0, -1)
    fisher = np.sum((n_tr * (mean_intra - mean_inter) ** 2), 2) / np.sum((n_tr * var_intra), 2)
    fisher = np.swapaxes(fisher, 0, 1)
    xlab = [int(hz) for hz in f]
    sns.heatmap(fisher, xticklabels=xlab, yticklabels=ylab, vmin=0, vmax=1)
    return fisher


def run_psd(s, h, flim, flat=False):
    fmin_bin = int(flim[0] / 2)
    fmax_bin = int(flim[1] / 2) + 1
    # s shape is (samples, channels, trials)
    n_chan = np.shape(s)[1]
    n_tr = np.shape(s)[2]
    s_psd = np.zeros((fmax_bin-fmin_bin, n_chan, n_tr))
    for tr in range(n_tr):
        for ch in range(n_chan):
            f, s_psd[:, ch, tr] = np.array(
                signal.periodogram(s[:, ch, tr], fs=h['SampleRate'], scaling='density'))[:, fmin_bin:fmax_bin]
    # s_psd shape is (freq, channels, trials)
    if flat:
        s_psd_flat = np.zeros((n_tr, n_chan * np.shape(s_psd)[0]))
        s_psd = np.swapaxes(s_psd, 0, 1)
        for tr in range(n_tr):
            s_psd_flat[tr, :] = s_psd[:, :, tr].flatten()
        return s_psd_flat  # s_psd_flat shape is (trials, channels*freq)
    return s_psd


def trial_psd(tr, fs, flim):
    fmin_bin = int(flim[0] / 2)
    fmax_bin = int(flim[1] / 2) + 1
    # tr shape is (samples, channels)
    n_chan = np.shape(tr)[1]
    t_psd = np.zeros((fmax_bin-fmin_bin, n_chan))
    for ch in range(n_chan):
        f, t_psd[:, ch] = np.array(
            signal.periodogram(tr[:, ch], fs=fs, scaling='density'))[:, fmin_bin:fmax_bin]
    # t_psd shape is (freq, channels)
    return t_psd


def rank_avg_fisher(run_fishers):
    ranks = np.zeros_like(run_fishers)
    for f, fish in enumerate(run_fishers):
        sh = np.shape(fish)
        rank = np.size(fish) - scipy.stats.rankdata(fish, 'min') + 1
        ranks[f, :, :] = np.reshape(np.ones_like(rank) / rank, sh)
    return sum(np.multiply(ranks, run_fishers), 0)


def select_features(ranked_features, xlabs, ylabs, nfeat):
    sh = np.shape(ranked_features)
    rank = np.reshape((np.size(ranked_features) - scipy.stats.rankdata(ranked_features, 'min') + 1), sh)
    feat_mask = np.where(rank <= nfeat, 1, 0)  # (channels, freq)
    features = np.array([[y, x] for y in ylabs for x in xlabs])
    flatmap = np.array(range(np.shape(features)[0]))
    selected_features = flatmap.ravel()[np.flatnonzero(feat_mask)]
    selected_features = [list(features[v]) for v in selected_features]
    return feat_mask, selected_features


def build_training_data(run_psds, hs, mask):
    r = 0
    x = np.zeros((np.shape(run_psds[0])[-1] * len(run_psds), 20))
    y = np.zeros(np.shape(run_psds[0])[-1] * len(run_psds))
    for i, h in enumerate(hs):
        s = run_psds[:, :, i]
        for tr in range(np.shape(s)[0]):
            x[r, :] = s[tr, :].ravel()[np.flatnonzero(mask)]
            y[r] = h['Classlabel'][tr]
            r = r + 1
    return x, y


n_chan = 13
electrode = "Gel"
channel_path = "chaninfo_" + electrode
chaninfo = loadmat(channel_path + '.mat')[channel_path]['channels']
h1 = loadmat('h1.mat')['h']
h2 = loadmat('h2.mat')['h']
h3 = loadmat('h3.mat')['h']
h4 = loadmat('h4.mat')['h']
s1 = loadmat('s1.mat')['s'][:, :n_chan]
s2 = loadmat('s2.mat')['s'][:, :n_chan]
s3 = loadmat('s3.mat')['s'][:, :n_chan]
s4 = loadmat('s4.mat')['s'][:, :n_chan]

fs = h1['SampleRate']
broad = [4, 30]
broad_filt = ButterFilter(2, 'band', fs, broad)
runs = [s1, s2, s3, s4]
heads = [h1, h2, h3, h4]
plt.subplot(2, 3, 1)
i = 1
n_trials = len(h1['Classlabel'])
xlabels = [int(x) for x in range(broad[0], broad[1] + 2, 2)]
ylabels = [chaninfo[c]['labels'] for c, d in enumerate(chaninfo)]

# Feature Selection
car_filt = CARFilter(n_chan)
fishers = np.zeros((len(runs), n_chan, len(xlabels)))
for S, H in zip(runs, heads):
    S = broad_filt.apply_filter(S)
    s_split = list(runs2trials_split([S], [H], 0.5))
    s_split_filt = np.array([car_filt.apply_filter(s_j) for s_j in s_split])
    plt.subplot(2, 3, i)
    plt.margins(0, 0.1)
    plt.title("Run " + str(i))
    fishers[i - 1, :, :] = run_psd_fisher(s_split_filt, H, broad, ylabels)
    i = i + 1


plt.subplot(2, 3, 5)
plt.title("Rank weighted sum")
rank_sum_fisher = rank_avg_fisher(fishers)
sns.heatmap(rank_sum_fisher, xticklabels=xlabels, yticklabels=ylabels)

plt.subplot(2, 3, 6)
plt.title("log(Rank weighted sum)")
sns.heatmap(np.log10(rank_sum_fisher), xticklabels=xlabels, yticklabels=ylabels)
plt.show()

mask, feats = select_features(rank_sum_fisher, xlabels, ylabels, n_trials)

# Decoder Training
r = 0
unmasked_epochs = np.zeros((n_trials, np.size(mask), len(runs)))
for S, H in zip(runs, heads):
    S = broad_filt.apply_filter(S)
    s = runs2trials([S], [H], 0.5)
    s = car_filt.apply_filter(s)
    unmasked_epochs[:, :, r] = run_psd(s, H, broad, flat=True)
    r = r + 1
x, y = build_training_data(unmasked_epochs, heads, mask)
clf = LinearDiscriminantAnalysis()
clf.fit(x, y)



print("hello world")
