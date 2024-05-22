import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)  # Для воспроизводимости результатов
n = 20
x_values = np.linspace(-1.8, 2, n)
epsilons = np.random.normal(0, 1, n)
y_values = 4 + 3 * x_values + epsilons  # Эталонные значения

# Функция потерь по методу наименьших квадратов
def least_squares_loss(params, y_target):
    a, b = params
    return np.sum((y_target - b - a * x_values)**2)

# Функция потерь по методу наименьших модулей
def least_absolute_deviation_loss(params, y_target):
    a, b = params
    return np.sum(np.abs(y_target - b - a * x_values))

# Ищем МНК-оценки для метода наименьших квадратов
result_ls = minimize(least_squares_loss, [0, 0], args=(y_values,))
a_ls, b_ls = result_ls.x

# Ищем МНК-оценки для метода наименьших модулей
result_lad = minimize(least_absolute_deviation_loss, [0, 0], args=(y_values,))
a_lad, b_lad = result_lad.x

# Вывод результатов
print("Результаты для МНК:")
print(f"a = {a_ls.round(3)}, b = {b_ls.round(3)}")
print(f"\\delta a = {abs(4-a_ls.round(3))*50}, \\delta_b = {abs(3-b_ls.round(3))*50}")
print("Результаты для МНМ:")
print(f"a = {a_lad.round(3)}, b = {b_lad.round(3)}")
print(f"\\delta_a = {abs(4-a_lad.round(3))*50}, \\delta_b = {abs(3-b_lad.round(3))*50}")

x_interp = np.linspace(-2.5, 2.5, 25)
y_interp_ls, y_interp_lad = a_ls * x_interp + b_ls, a_lad * x_interp + b_lad

plt.scatter(x_values, y_values)
plt.plot(x_interp, y_interp_ls, label='МНК', color='r')
plt.plot(x_interp, y_interp_lad, label='МНМ', color='y')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Внесём возмущения
y_perturbed = y_values.copy()
y_perturbed[0] += 10
y_perturbed[-1] -= 10

# Ищем МНК-оценки для изменённой выборки методом наименьших квадратов
result_ls_perturbed = minimize(least_squares_loss, [0, 0], args=(y_perturbed,))
a_ls_perturbed, b_ls_perturbed = result_ls_perturbed.x

# Ищем МНК-оценки для изменённой выборки методом наименьших модулей
result_lad_perturbed = minimize(least_absolute_deviation_loss, [0, 0], args=(y_perturbed, ))
a_lad_perturbed, b_lad_perturbed = result_lad_perturbed.x

x_interp_pert = np.linspace(-2.5, 2.5, 25)
y_interp_ls_pert, y_interp_lad_pert = a_ls_perturbed * x_interp_pert + b_ls_perturbed, a_lad_perturbed * x_interp_pert + b_lad_perturbed

plt.scatter(x_values, y_perturbed)
plt.plot(x_interp_pert, y_interp_ls_pert, label='МНК', color='r')
plt.plot(x_interp_pert, y_interp_lad_pert, label='МНМ', color='y')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Вывод результатов для изменённой выборки
print("\nРезультаты для изменённой выборки:")
print("Результаты для МНК:")
print(f"a = {a_ls_perturbed.round(3)}, b = {b_ls_perturbed.round(3)}")
print(f"\\delta_a = {abs(4-a_ls_perturbed.round(3))*50}, \\delta_b = {abs(3-b_ls_perturbed.round(3))*50}")
print("Результаты для МНМ:")
print(f"a = {a_lad_perturbed.round(3)}, b = {b_lad_perturbed.round(3)}")
print(f"\\delta_a = {abs(4-a_lad_perturbed.round(3))*50}, \\delta_b = {abs(3-b_lad_perturbed.round(3))*50}")
