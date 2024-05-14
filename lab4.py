import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

sample_sizes = [20, 100]
discr = {20: 10, 100: 32}
distributions = {
    "Normal": (np.random.normal, {"loc": 0, "scale": 1}),
    "Uniform": (np.random.random, {}),
}

class RandomSample:

    def get_sample(self, dist_func, size, **dist_params):
        return dist_func(size=size, **dist_params)

    def normal_mu_bounds(self, significance):
        size = len(self.sample)
        mean, std = np.mean(self.sample), np.std(self.sample)
        quantile = sts.t.ppf(1 - significance / 2, size - 1)
        return mean - std * quantile / np.sqrt(size - 1), mean + std * quantile / np.sqrt(size - 1)

    def normal_sigma_bounds(self, significance):
        size = len(self.sample)
        std = np.std(self.sample)
        return  (std * np.sqrt(size) )/ np.sqrt(sts.chi2.ppf(1-significance / 2, size - 1)), \
        (std * np.sqrt(size)) / np.sqrt(sts.chi2.ppf(significance / 2, size - 1))

    def uniform_mu_bounds(self, significance):
        size = len(self.sample)
        mean=np.mean(self.sample)
        std = np.std(self.sample)
        quantile = sts.norm.ppf(1-significance/2)

        return mean - (std * quantile / np.sqrt(size - 1)), mean + std * quantile / np.sqrt(size - 1)

    def uniform_sigma_bounds(self, significance):
        std= np.std(self.sample)
        size = len(self.sample)

        excess = ( np.sum((self.sample - np.mean(self.sample))**4) / size) / std**4 - 3
        return std * (1-0.5*sts.norm.ppf(1-significance/2) * np.sqrt((excess + 2) / size)), std * (1+0.5*sts.norm.ppf(1-significance/2) * np.sqrt((excess + 2) / size))

normal_sigmas, uniform_sigmas = [], []

def show_info(size):
    significance = 0.7
    rs = RandomSample()
    rs.sample = rs.get_sample(dist_func, size, **dist_params)

    if dist_name == "Normal":
        mu_min, mu_max = rs.normal_mu_bounds(significance)
        sig_min, sig_max = rs.normal_sigma_bounds(significance)
        normal_sigmas.append((sig_min, sig_max))
    if dist_name == "Uniform":
        mu_min, mu_max = rs.uniform_mu_bounds(significance)
        sig_min, sig_max = rs.uniform_sigma_bounds(significance)
        uniform_sigmas.append((sig_min, sig_max))

    print(f"""{dist_name} Distribution: size = {size}
    {mu_min} < Mu < {mu_max}
    {sig_min} < Sigma < {sig_max}\n 
    """)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dist_name}. N = {size}")
    labels = [
        'Histogram', r'$\mu_{\min}$', r'$\mu_{\max}$',
        r'$\mu_{\min} - \sigma_{\max}$', r'$\mu_{\max} + \sigma_{\max}$'
    ]
    plt.hist(rs.sample, bins=discr[size], density=True, color='green')
    color_table = {0: 'r', 1: 'b'}
    params = [mu_min, mu_max, mu_min - sig_max, mu_max + sig_max]
    for i, param in enumerate(params):
        plt.axvline(param, color=color_table[i // 2], linestyle='-', linewidth=3, marker='o')
    plt.show()

def plot_sigmas(sigmas):
    plt.hlines(0.5, xmin=sigmas[0][0], xmax=sigmas[0][1], color='b', linestyles='-')
    plt.hlines(0.7, xmin=sigmas[1][0], xmax=sigmas[1][1], color='r', linestyles='-')
    plt.plot([sigmas[0][0], sigmas[0][1]], [0.5, 0.5], 'ro', markersize=5)
    plt.plot([sigmas[1][0], sigmas[1][1]], [0.7, 0.7], 'ro', markersize=5)
    plt.legend(labels=[r'$N=20: [\sigma_{\min}, \sigma_{\max}]$',
                       r'$N=100: [\sigma_{\min}, \sigma_{\max}]$'
                       ])
    plt.show()

for dist_name, (dist_func, dist_params) in distributions.items():
    for i, size in enumerate(sample_sizes):
        show_info(size)

plot_sigmas(normal_sigmas)
plot_sigmas(uniform_sigmas)