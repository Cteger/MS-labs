import math
import enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

data_initial = np.sort(np.loadtxt("C:/Users/user/source/repos/MatStat/lab7/lab7/0.15V_sp298.txt")).T
data = np.array([])
for inner in data_initial:
    if len(data) == 0:
        data = inner
    else:
        data = np.concatenate((data, inner))
del data_initial

plt.plot(data)
plt.show()

data = medfilt(data, kernel_size=9)

bound_idcs = [1023, 1024, 9215]
segments = []
start_idx = 0
for end_idx in bound_idcs:
    segment = data[start_idx:end_idx]
    segments.append(segment)
    start_idx = end_idx

plt.plot(np.arange(0, len(data)), data, color='y')
colors_n_labels = [('b', 'Background'), ('y', 'Transition'), ('r', 'Signal')]
for i in range(len(segments)):
    if i == 0:
        plt.plot(np.arange(i, bound_idcs[i]), segments[0], color=colors_n_labels[0][0], label=colors_n_labels[0][1])
    else:
        plt.plot(np.arange(bound_idcs[i-1], bound_idcs[i]), segments[i], color=colors_n_labels[i][0], label=colors_n_labels[i][1])
plt.legend()
plt.show()

class SegmentType(enum.Enum):
    Background = 1
    Transition = 2
    Signal = 3

class SegmentAnalyzer:
    def __init__(self, sample, type):
        self.sample = sample
        self.type = type
        self.k = round(1 + math.log2(len(sample)))
        self.s_in, self.s_out, self.F = None, None, None

    def analyze(self):
        widehat = np.mean(self.sample)

        def split_segments():
            n = len(self.sample)
            segment_size = n // self.k
            remainder = n % self.k
            segment_sizes = [segment_size] * self.k
            for i in range(remainder):
                segment_sizes[i] += 1
            segments, start = [], 0
            for size in segment_sizes:
                segments.append(self.sample[start:start + size])
                start += size
            return segments

        parts = split_segments()

        antinan_wrapper = lambda x: x if not np.isnan(x) else 0

        def s_in():
            def s_in_i(part):
                result = sum([(part[j] - widehat) ** 2 for j in range(len(part))]) / (self.k*(self.k-1))
                return antinan_wrapper(result)

            res = sum([s_in_i(part) for part in parts])
            print(" & ".join(map(str, [int(s_in_i(part)) for part in parts] )) + "\\\\")
            return antinan_wrapper(res)

        def s_out():
            avgs = np.array([np.mean(part) for part in parts])
            avg = np.mean(avgs)

            def s_out_i(avgs_part):
                result = (avgs_part - avg) ** 2 * self.k / (self.k - 1)
                return antinan_wrapper(result)

            result = sum([(avgs_part - avg) ** 2 for avgs_part in avgs])
            print(" & ".join(map(str, [int(s_out_i(avgs_part)) for avgs_part in avgs]) ) + "\\\\")
            return antinan_wrapper(result)

        def F_print():
            type_text = "Background" if self.type == SegmentType.Background \
                else "Transition" if self.type == SegmentType.Transition \
                else "Signal"
            print(f"{type_text} & {self.k} & {s_in()} & {s_out()} & {antinan_wrapper(s_out() / s_in())} \\\\ \n")

        F_print()

segment_types = [SegmentType.Background, SegmentType.Transition, SegmentType.Signal]
for idx, segment in enumerate(segments):
    segm_inst = SegmentAnalyzer(segment, segment_types[idx])
    segm_inst.analyze()
