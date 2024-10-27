import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

energies, counts = [], []

with open("hist2.csv", "r") as f:
    for line in f.readlines():
        _e, _c = [float(i) for i in line.split(',')]
        energies.append(_e)
        counts.append(_c)

x = np.array(energies)
y = np.array(counts)

counts, bin_edges = np.histogram(x, weights=y, bins=45)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
second_derivative = np.gradient(np.gradient(counts))
minimum = np.argmin(second_derivative)
split = bin_centers[minimum]

left_x = x[x < split]
left_y = y[:len(left_x)]
right_x = x[x >= split]
right_y = y[len(left_x):]

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

left_initial = [max(left_y), np.mean(left_x), np.std(left_x)]
popt_left, _ = curve_fit(gaussian, left_x, left_y, p0=left_initial)
right_initial = [max(right_y), np.mean(right_x), np.std(right_x)]
popt_right, _ = curve_fit(gaussian, right_x, right_y, p0=right_initial)

x_fit = np.linspace(min(x), max(x), 10000)
y_left_fit = gaussian(x_fit, *popt_left)
y_right_fit = gaussian(x_fit, *popt_right)

area1 = trapezoid(y_left_fit, x_fit)
area2 = trapezoid(y_right_fit, x_fit)
ratio = area1/area2

plt.hist(x, weights=y, bins=45, color='#FCE7AE', label='Histogram')
plt.plot(x_fit, y_left_fit, color='#C09BBC', label='Particle A')
plt.plot(x_fit, y_right_fit, color='#78C3C9', label='Particle B')
plt.text(0.63, 0.74, f"Particle A Area: {area1:.2f}", transform=plt.gca().transAxes)
plt.text(0.63, 0.69, f"Particle B Area: {area2:.2f}", transform=plt.gca().transAxes)
plt.text(0.63, 0.64, f"Ratio [A:B]: {ratio:.2f}", transform=plt.gca().transAxes)
plt.xlabel('Energy')
plt.ylabel('Counts')
plt.title('Energy Spectrum')
plt.legend()
plt.grid(True)
plt.show()