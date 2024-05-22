import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse

# Function to generate random samples
def sample_generator(size, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, size)

# Function to calculate quadrant correlation coefficient
def quadrant_correlation_coefficient(x, y):
    return np.mean(np.sign(x) == np.sign(y))

# Function to calculate mean of squares
def mean_of_squares(data):
    return np.mean(np.array(data) ** 2)

# Function to compute variance
def compute_variance(data):
    return np.mean((data - np.mean(data))**2)

# Parameters
sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
num_samples = 1000

# Visualize sample and equal probability ellipse
def plot_ellipse(data, title):
    fig, axs = plt.subplots(len(sizes), len(rhos), figsize=(18, 12))
    for i, size in enumerate(sizes):
        for j, rho in enumerate(rhos):
            samples = data[(size, rho)]
            ax = axs[i, j]
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
            covar = np.cov(samples.T)
            mean = np.mean(samples, axis=0)
            ellipse = Ellipse(mean, width=2 * np.sqrt(covar[0, 0]), height=2 * np.sqrt(covar[1, 1]), angle=0, edgecolor='r', linestyle='--', fill=False)
            ax.add_patch(ellipse)
            ax.set_title(f'Size={size}, Rho={rho}')
            ax.grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Output information
def show_results(data, name):
    print(f"""
    {name}:
        Mean: {np.mean(data).round(3)}
        Mean of Squares: {mean_of_squares(data).round(3)}
        Variance: {compute_variance(data).round(3)}
""")

# Calculate results for all distributions
distributions = {}
for size in sizes:
    for rho in rhos:
        print(f"Size: {size}, Rho: {rho}")
        pearson_corrs, spearman_corrs, quadrant_corrs = [], [], []

        for _ in range(num_samples):
            samples = sample_generator(size, rho)
            x = samples[:, 0]
            y = samples[:, 1]

            pearson_corr, _ = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)
            quadrant_corr = quadrant_correlation_coefficient(x, y)

            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            quadrant_corrs.append(quadrant_corr)

        # Store the samples for visualization
        distributions[(size, rho)] = samples

        show_results(pearson_corrs, "Pearson")
        show_results(quadrant_corrs, "Quadrant")
        show_results(spearman_corrs, "Spearman")

# Visualize all distributions in one figure
plot_ellipse(distributions, '')

# Generating mixed samples
def mixed_sample_generator(size):
    return 0.9 * sample_generator(size, 0.9) + 0.1 * sample_generator(size, -0.9)

# Calculate mixed distribution
mixed_distributions = {}
for size_ in sizes:
    print(f"Mixed Distribution\nSize: {size_}\n")
    pearson_corrs, spearman_corrs, quadrant_corrs = [], [], []

    for _ in range(num_samples):
        mix_samples = mixed_sample_generator(size_)
        x = mix_samples[:, 0]
        y = mix_samples[:, 1]

        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)
        quadrant_corr = quadrant_correlation_coefficient(x, y)

        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        quadrant_corrs.append(quadrant_corr)

    # Store the mixed samples for visualization
    mixed_distributions[size_] = mix_samples

    show_results(pearson_corrs, "Pearson")
    show_results(quadrant_corrs, "Quadrant")
    show_results(spearman_corrs, "Spearman")


# Display all combined distributions in a single diagram
figure, axes = plt.subplots(1, len(sizes), figsize=(18, 6))
for index, size_value in enumerate(sizes):
    combined_samples = mixed_distributions[size_value]
    ax_ = axes[index]
    ax_.scatter(combined_samples[:, 0], combined_samples[:, 1], alpha=0.5)
    create_ellipse(ax_, np.cov(combined_samples.T), np.mean(combined_samples, axis=0), edgecolor='r', linestyle='--', fill=False)
    ax_.set_title(f'Combined N={size_value}')
    ax_.grid(True)
figure.suptitle('')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()