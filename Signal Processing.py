import scipy.stats
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import math
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os

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

    def noncausal_filter(self, raw_sig):
        b, a = signal.butter(self.n, self.cutoff, self.btype, fs=self.fs)
        return signal.filtfilt(b, a, raw_sig, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

    def causal_filter(self, raw_sig):
        b, a = signal.butter(self.n*2, self.cutoff, self.btype, fs=self.fs)
        return signal.lfilter(b, a, raw_sig, axis=0)

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

    def apply_filter(self, s, rawflag):
        if rawflag:  # raw data (samples, channels)
            return super().filter_raw_sig(s, self.filter)
        else:  # trial data (win, channels, trials)
            for tr in range(np.shape(s)[0]):
                s[tr] = np.matmul(s[tr], self.filter)
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


def runs2trials_split(ss, hs):
    lt = 0
    rt = 0
    s_L = []
    s_R = []
    for s, h in zip(ss, hs):
        for i, t in enumerate(h['EVENT']['TYP']):
            if t == 1000 and h['EVENT']['TYP'][i+4] in [7692, 7693]:
                s_L.append(s[h['EVENT']['POS'][i]:h['EVENT']['POS'][i+4], :])
                lt = lt + 1
            elif t == 1000 and h['EVENT']['TYP'][i+4] in [7702, 7703]:
                s_R.append(s[h['EVENT']['POS'][i]:h['EVENT']['POS'][i+4], :])
                rt = rt + 1
    return s_L, s_R


def runs2trials(ss, hs):
    trs = []
    truths = []
    for s, h in zip(ss, hs):
        triggers = h['EVENT']['TYP']
        starts = [h['EVENT']['POS'][i] for i, v in enumerate(triggers) if v == 1000]
        ends = [h['EVENT']['POS'][i] for i, v in enumerate(triggers) if v in [7692, 7693, 7702, 7703]]
        t = 0
        for st, en in zip(starts, ends):
            trs.append(s[st:en, :])  # trs shape is (trials, samples, channels)
            truths.append(h['Classlabel'][t])
            t = t + 1
    return trs, truths


def run_psd_fisher(s, h, flim, ylab):
    fmin_bin = int(flim[0] / 2)
    fmax_bin = int(flim[1] / 2) + 1
    c = np.shape(s)[0]
    n_chan = 13
    n_tr = np.shape(s)[1]
    s_psd = np.zeros((fmax_bin-fmin_bin, n_chan, n_tr, c))
    for j in range(c):
        for tr in range(n_tr):
            for ch in range(n_chan):
                f, s_psd[:, ch, tr, j] = np.array(
                    signal.periodogram(s[j, tr][:, ch], fs=h['SampleRate'], nfft=256, scaling='density'))[:,fmin_bin:fmax_bin]
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
    # s shape is (trials, samples, channels)
    n_chan = np.shape(s[0])[1]
    n_tr = np.shape(s)[0]
    s_psd = np.zeros((fmax_bin-fmin_bin, n_chan, n_tr))
    for tr in range(n_tr):
        for ch in range(n_chan):
            f, s_psd[:, ch, tr] = np.array(
                signal.periodogram(s[tr][:, ch], fs=h['SampleRate'], nfft=256, scaling='density'))[:, fmin_bin:fmax_bin]
    # s_psd shape is (freq, channels, trials)
    if flat:
        s_psd_flat = np.zeros((n_tr, n_chan * np.shape(s_psd)[0]))
        s_psd = np.swapaxes(s_psd, 0, 1)
        for tr in range(n_tr):
            s_psd_flat[tr, :] = s_psd[:, :, tr].flatten()
        return s_psd_flat  # s_psd_flat shape is (trials, channels*(freq))
    return np.swapaxes(s_psd, 0, 1)


def trial_psd(tr, fs, flim):
    fmin_bin = int(flim[0] / 2)
    fmax_bin = int(flim[1] / 2) + 1
    # tr shape is (samples, channels)
    n_chan = 13
    t_psd = np.zeros((n_chan, fmax_bin-fmin_bin))
    for ch in range(n_chan):
        f, t_psd[ch, :] = np.array(
            signal.periodogram(tr[:, ch], fs=fs, nfft=256, scaling='density'))[:, fmin_bin:fmax_bin]
    # t_psd shape is (channels, freq)
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


def build_training_data(runs, hs, win, lap, fs, t_filt, sp_filt, flim, mask):
    win = math.floor(win * fs)  # seconds -> samples
    step = win - math.floor(lap * fs)  # seconds -> samples
    x = np.zeros(np.sum(mask))
    y = []
    for s, h in zip(runs, hs):
        for tr, trial in enumerate(s):
            i = 0
            while (i + step < np.shape(trial)[0]):  # loop through windows
                sample = trial[i:i + win, :]
                sample = t_filt.noncausal_filter(sample)
                sample = sp_filt.apply_filter(sample, True)
                sample_psd = trial_psd(sample, fs, flim)
                x = np.row_stack((x, sample_psd.ravel()[np.flatnonzero(mask)]))
                y.append(h['Classlabel'][tr])
                i = i + step
    return x[1:, :], y


def simulate_trial(trial, win, lap, fs, t_filt, sp_filt, flim, mask, clf, g_truth, thresh):
    win = math.floor(win*fs)  # seconds -> samples
    step = win - math.floor(lap*fs)  # seconds -> samples
    accum_prob = [0.5]
    BCI_prob = [0.5]
    alpha = 0.9
    i = 0
    while(i+step<len(trial)):  # loop through windows
        sample = trial[i:i+win]
        sample_filt = t_filt.noncausal_filter(sample)
        sample_filt = sp_filt.apply_filter(sample_filt, True)
        sample_psd = trial_psd(sample_filt, fs, flim)
        sample_feats = np.array(sample_psd.ravel()[np.flatnonzero(mask)]).reshape(1, -1)
        prob_both = clf.predict_proba(sample_feats)  # put in classifier
        print("Probs: ", prob_both)
        BCI_prob.append(prob_both[0, g_truth-1])
        # Calculate sample sample level performance - Satvik
        accum_prob.append(alpha*accum_prob[-1] + (1-alpha)*BCI_prob[-1])  # accumulate evidence
        if accum_prob[-1] > thresh[g_truth-1]:  # compare to correct class threshold
            print("accum Prob:", accum_prob, "correct: ", g_truth)
            return {"BCI_probs": BCI_prob, "accum_probs": accum_prob, "decision": g_truth, "correct": 1}
        elif accum_prob[-1] < 1-thresh[g_truth % 2]:  # compare to incorrect class threshold
            print("accum Prob:", accum_prob,"incorrect: ", g_truth)
            return {"BCI_probs": BCI_prob, "accum_probs": accum_prob, "decision": (g_truth % 2)+1, "correct": 0}
        i = i+step
    print("accum Prob:", accum_prob, "no decision: ", g_truth)
    return {"BCI_probs": BCI_prob, "accum_probs": accum_prob, "decision": float("nan"), "correct": 0}


def cross_val(clf, x, y,  folds, gs=False):

    cv = KFold(folds, shuffle=False)
    cv_results = cross_val_score(clf, x, y, scoring='accuracy', cv=cv)
    cv_mean = cv_results.mean()

    print("Mean Cross Validation Score Across Folds: ", cv_mean)

    # Hyperparameter Optimization
    if gs:
        p_grid = {"solver": ['svd'], "tol": [0.0001, 0.0002, 0.0003], "store_covariance": [True, False]}
        gs_clf = GridSearchCV(clf, param_grid=p_grid, cv=cv, scoring='accuracy', verbose=2)
        gs_clf.fit(x, y)
        clf_params = gs_clf.best_params_
        # Apply best params to classifier
        return clf_params
    
    return clf


# File Structure
# > subject_<id>
#     > gel
#        > offline
#            > session_1
#                > h1.mat
#                > s1.mat
#        > online
#            > session_1
#                > h1.mat
#                > s1.mat
#            > session_2
#                > h1.mat
#                > s1.mat
#     > poly
#        > offline
#            > session_1
#                > h1.mat
#                > s1.mat
#        > online
#            > session_1
#                > h1.mat
#                > s1.mat
#            > session_2
#                > h1.mat
#                > s1.mat

# Load Data Parameters
subject = 5
electrode = 'Poly'
session_type = 'offline'
session_id = 1
n_chan = 13

#### SET THIS TO FALSE IF YOU WANT TO JUST USE DEFAULT VALUES SET ABOVE #####
take_inputs = False

# Take parsed inputs
if take_inputs:
    print(">>> Leave blank to use default values")
    inputin = input('subject_id: ')
    if inputin != '':
        subject = int(inputin)

    inputin = input('Electrode type: ')
    if inputin != '':
        if(inputin[0] == 'g'):
            electrode = 'Gel'
        if(inputin[0]=='d'):
            electrode = 'Dry'

    inputin = input('Session id: ')
    if inputin != '':
        session_id = int(inputin)

    if(session_id < 1 or session_id > 2):
        print("Incorrect session_id")
        exit()
    if(subject < 4 or subject > 6):
        print("invalid subject")
        exit()

file_path = "subject_" + str(subject) + "/" + electrode + "/" + session_type + "/session_" + str(session_id) + "/"

# os.chdir('/Users/satvik/Desktop/BCI_Motor_Imagery/')

channel_path = 'chaninfo_' + electrode
chaninfo = loadmat(channel_path + '.mat')[channel_path]['channels']

# Signal data loading based on selected inputs - uncomment once data exists
h1 = loadmat(file_path + 'h1.mat')['h']
h2 = loadmat(file_path + 'h2.mat')['h']
h3 = loadmat(file_path + 'h3.mat')['h']
h4 = loadmat(file_path + 'h4.mat')['h']

s1 = loadmat(file_path + 's1.mat')['s'][:, :n_chan]
s2 = loadmat(file_path + 's2.mat')['s'][:, :n_chan]
s3 = loadmat(file_path + 's3.mat')['s'][:, :n_chan]
s4 = loadmat(file_path + 's4.mat')['s'][:, :n_chan]


fs = h1['SampleRate']
broad = [4, 30]
broad_filt = ButterFilter(2, 'band', fs, broad)

runs = [s1, s2, s3, s4]
heads = [h1, h2, h3, h4]


n_trials = len(h1['Classlabel'])
xlabels = [int(x) for x in range(broad[0], broad[1] + 2, 2)]
ylabels = [chaninfo[c]['labels'] for c, d in enumerate(chaninfo)]
plt.subplot(2, 3, 1)
i = 1

# Feature Selection
car_filt = CARFilter(n_chan)
fishers = np.zeros((len(runs), n_chan, len(xlabels)))
for S, H in zip(runs, heads):
    s_temp = broad_filt.causal_filter(S)
    s_split = list(runs2trials_split([s_temp], [H]))
    s_split_filt = np.array([car_filt.apply_filter(s_j, False) for s_j in s_split])
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

mask, feats = select_features(rank_sum_fisher, xlabels, ylabels, 20)
print(feats)

# Decoder Training
unmasked_epochs = []
for S, H in zip(runs, heads):
    #S = broad_filt.noncausal_filter(S)
    s, truths = runs2trials([S], [H])
    #s = car_filt.apply_filter(s, False)
    unmasked_epochs.append(s)
x, y = build_training_data(unmasked_epochs, heads, 1, 0.1, fs, broad_filt, car_filt, broad, mask)
clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
clf.fit(x, y)
print(clf.predict_proba(x))




# Cross Validation - Satvik

val = cross_val(clf, x, y, folds=4, gs=False)


# Online Simulation
win = 1  # 1 s
lap = 0.1  # 100 ms
thresh = [0.6, 0.6]  # left, right or class 1, class 2
# collect all trials and ground truths

session_type = 'online'
session_id = 1
file_path = "subject_" + str(subject) + "/" + electrode + "/" + session_type + "/session_" + str(session_id) + "/"
h1_on = loadmat(file_path + 'h1.mat')['h']
h2_on = loadmat(file_path + 'h2.mat')['h']
h3_on = loadmat(file_path + 'h3.mat')['h']

s1_on = loadmat(file_path + 's1.mat')['s'][:, :n_chan]
s2_on = loadmat(file_path + 's2.mat')['s'][:, :n_chan]
s3_on = loadmat(file_path + 's3.mat')['s'][:, :n_chan]

# For each trial
online_runs = [s1_on, s2_on, s3_on]
online_heads = [h1_on, h2_on, h3_on]
online_trials, online_truths = runs2trials(online_runs, online_heads)
plt.figure()
plt.title("Probabilistic Decision Making")
plt.xlabel("Sample")
plt.ylabel("Correct Class Probability")
plt.axhline(thresh[0])
plt.axhline(1-thresh[0])
endx = 0
startx = 0
for tr, g_truth in zip(online_trials, online_truths):
    decision = simulate_trial(tr, win, lap, fs, broad_filt, car_filt, broad, mask, clf, g_truth, thresh)
    endx = startx + len(decision['accum_probs'])
    xvals = range(startx, endx)
    plt.scatter(xvals, decision['accum_probs'], s=1, c='b')
    plt.scatter(xvals, decision['BCI_probs'], s=1, c='r')
    plt.axvline(endx, linewidth=0.5, color='g')
    startx = endx + 1
plt.show()
# compute trial performance

print("hello world")