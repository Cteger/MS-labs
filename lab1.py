import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ��������� �������
sample_sizes = [10, 50, 1000]

# ���������� �������������
norm_samples = [np.random.normal(0, 1, size) for size in sample_sizes]

# ������������� ����
cauchy_samples = [np.random.standard_cauchy(size) for size in sample_sizes]

# ������������� ���������
t_samples = [np.random.standard_t(3, size) for size in sample_sizes]

# ������������� ��������
poisson_samples = [np.random.poisson(10, size) for size in sample_sizes]

# ����������� �������������
uniform_samples = [np.random.uniform(-np.sqrt(3), np.sqrt(3), size) for size in sample_sizes]

# ���������� �������� � ����������
distributions = [
    (norm_samples, "Normal Distribution (N(0, 1))"),
    (cauchy_samples, "Cauchy Distribution (C(0, 1))"),
    (t_samples, "Student's t-Distribution (t(0, 3))"),
    (poisson_samples, "Poisson Distribution (P(10))"),
    (uniform_samples, "Uniform Distribution (U(-sqrt(3), sqrt(3))")
]

for samples, title in distributions:
    for i, sample in enumerate(samples):
        plt.figure(figsize=(10, 5))
        
        # ���������� �����������
        plt.plot()
        plt.hist(sample, bins=30, density=True, alpha=0.5, color='g', edgecolor='black')
        plt.title(f'Histogram of {title} (Sample size: {sample_sizes[i]})')
        
        # ���������� ������� ��������� �������������
        #plt.subplot(1, 2, 2)
        min_x = np.min(sample)
        max_x = np.max(sample)
        x = np.linspace(min_x, max_x, 1000)
        if i == 4:
            y = stats.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3))
        else:
            y = stats.gaussian_kde(sample).evaluate(x)
        plt.plot(x, y, 'r')
        plt.title(f'Density Plot of {title}')
        
        plt.show()
