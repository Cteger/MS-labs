import numpy as np

distributions = [
    ("normal", lambda x: np.random.normal(0, 1, x)),
    ("cauchy", lambda x: np.random.standard_cauchy(x)),
    ("student", lambda x: np.random.standard_t(3, x)),
    ("puasson", lambda x: np.random.poisson(10, x)),
    ("uniform", lambda x: np.random.uniform(-np.sqrt(3), np.sqrt(3), x))
]
sizes = [10, 100, 1000]
iterations = 1000

def analyze(data):
    def get_mean(data):
        return sum(data) / len(data)
    def get_median(data):
        return data[len(data) // 2] if len(data) % 2 == 1 else 0.5 * (data[len(data) // 2] + data[len(data) // 2 + 1])
    def get_halfsum_extreme(data):
        return 0.5 * (data[0] + data[-1])
    def get_halfsum_quartiles(data):
        return 0.5 * (data[len(data) // 4 - 1] + data[3 * len(data) // 4 - 1]) / 2 if len(data) % 4 == 0 else 0.5 * (data[len(data) // 4] + data[3 * len(data) // 4]) / 2
    def get_trimmed_mean(sample):
        r = round(len(sample) / 4)
        return sum(sample[r:len(sample) - r + 1]) / 4

    return (get_mean(data), get_median(data), get_halfsum_extreme(data), get_halfsum_quartiles(data), get_trimmed_mean(data))

def get_means(samples):
    transposed = [[row[i] for row in samples] for i in range(len(samples[0]))]
    return tuple([sum(transposed[i]) / len(transposed[i]) for i in range(len(transposed))])

def get_dispersion(samples):
    t_means = [[row[i] for row in samples] for i in range(len(samples[0]))]
    t_squares = [[row[i] ** 2 for row in samples] for i in range(len(samples[0]))]

    def maker_func(data):
        return sum(data[1]) / len(data[1]) - (sum(data[0]) / len(data[0])) ** 2

    return tuple([maker_func(tuple([t_means[i], t_squares[i]])) for i in range(len(t_means))])


for name, func in distributions:
    print("Statistics for " + name + " distribution:\n")
    for size in sizes:
        data = []
        for _ in range(iterations):
            data.append(analyze(func(size)))
        mean = get_means(data)
        dispersion = get_dispersion(data)

        print(f"""
    Size = {size}:
    Means:
    {' & '.join(['{:.2f}'.format(i) for i in mean])} 
    Dispersion:
    {' & '.join(['{:.2f}'.format(i) for i in dispersion])}

    """)