import numpy as np
import matplotlib.pyplot as plt
import math

import enum

class FuncType(enum.Enum):
    Gauss = 1
    Student = 2
    Uniform = 3

ft_hasher_s = {1 : FuncType.Gauss, 2 : FuncType.Student, 3 : FuncType.Uniform}
a_0, a_k = -1.5, 1.5

class Distr:
    def __init__(self, n, type):
        self.size, self.type = n, None
        if type in [1, 2, 3]:
            self.type = ft_hasher_s[type]
        def TableMaker():
            match self.type:
                case FuncType.Gauss: return np.random.normal(0, 1, self.size)
                case FuncType.Student: return np.random.standard_t(3, self.size)
                case FuncType.Uniform: return np.random.uniform(-np.sqrt(3), np.sqrt(3), self.size)
        self.distr = TableMaker()

    def DistrAlg(self):
        alpha = 0.0045#float(input("Input the alpha value: "))

        #k = round(1+math.log2(self.size))
        k=round(1+math.log2(self.size))
        #k += k % 2
        quantile = np.percentile(np.random.chisquare(df=k - 1, size=100000), 100 * (1 - alpha))

        #Разбиение
        class Chunk:
            def __init__(self, lb, rb):
                self.lb, self.rb = lb, rb
                self.nums = []

            def compare(self, num): return self.lb < num <= self.rb

        chunks = []
        step = 3 / (k-2)
        chunks.append(Chunk(-float('inf'), a_0))
        for i in range(k-2):
            chunks.append(Chunk(a_0 + i * step, a_0 + (i+1) * step))
        chunks.append(Chunk(a_k, float('inf')))

        for i in range(len(self.distr)):
            for j in range(len(chunks)):
                if chunks[j].compare(self.distr[i]):
                    chunks[j].nums.append(self.distr[i])
                    break

        #Подсчет вероятностей (p_i)
        from scipy.stats import norm, t
        probs = []
        for chunk in chunks:
            prob = None
            match self.type:
                case FuncType.Gauss : prob = norm.cdf(chunk.rb, loc=0, scale=1)-norm.cdf(chunk.lb, loc=0, scale=1)
                case FuncType.Student : prob = t.cdf(chunk.rb, 3)-t.cdf(chunk.lb, 3)
                case FuncType.Uniform :
                    lp = 0 if chunk.lb < -np.sqrt(3) else (chunk.lb + np.sqrt(3)) / (2 * np.sqrt(3))
                    rp = 1 if chunk.rb > np.sqrt(3) else (chunk.rb + np.sqrt(3)) / (2 * np.sqrt(3))
                    prob = rp - lp
            probs.append(prob)

        freqs = []
        for chunk in chunks:
            freq = len(chunk.nums) / self.size
            freqs.append(freq)

        def TablePrinter():

            print("\\hline \n $i$ & $\\delta_i$ & $p_i$ & $n_i$ & $(n_i-n*p_i)^2$ & $\chi_i^2$ \\\\n \\hline")
            for i in range(len(chunks)):
                print(f"{i+1} & ({round(chunks[i].lb, 3)};{round(chunks[i].rb, 3)}] & {round(probs[i], 3)} & {round(freqs[i], 3)} & {round((freqs[i]-self.size*probs[i])**2, 3)} & {round((freqs[i]-self.size*probs[i])**2 / probs[i] / self.size, 3)} \\\\")
            print("\\hline")

        TablePrinter()

        #Подсчет статистики хи-квадрат
        chi_stat = sum([(freqs[i]-self.size*probs[i])*(freqs[i]-self.size*probs[i])/(probs[i]) for i in range(len(probs))])/self.size
        name = "$\\mathcal{N}(0; 1)$" if self.type == FuncType.Gauss else "$\\mathcal{t}(3)$" if self.type == FuncType.Student else "$\\mathcal{U}(-\\sqrt{3}; \\sqrt{3})$"
        print(f"{name} Distribution, $N={self.size}$: $\\widehat(\\chi^2)$={round(chi_stat, 3)}, $\\chi_(1-\\alpha)^2$={round(quantile, 3)} \n")

sizes = [20, 100]
functypes = [1, 2, 3]

for size in sizes:
    for functype in functypes:
        distr = Distr(n=size, type=functype)
        distr.DistrAlg()
