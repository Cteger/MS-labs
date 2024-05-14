import numpy as np
import matplotlib.pyplot as plt

sample_sizes = [20, 100]

distributions = [
    ("Normal", lambda size: np.random.normal(0, 1, size)),
    ("Cauchy", lambda size: np.random.standard_cauchy(size)),
    ("Student", lambda size: np.random.standard_t(3, size)),
    ("Poisson", lambda size: np.random.poisson(10, size)),
    ("Uniform", lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size))
]

for distribution_name, distribution_func in distributions:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    i = 0
    for sample_size in sample_sizes:
        data = distribution_func(sample_size)
        axes[i].boxplot(data, vert=False)
        axes[i].set_title(f"{distribution_name}, n={sample_size}")
        i += 1
    plt.tight_layout()
    plt.show()