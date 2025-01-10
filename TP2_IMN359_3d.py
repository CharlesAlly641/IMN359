import numpy as np
import matplotlib.pyplot as plt


ordre_max = 100
t = np.linspace(-20, 20, 1000)

#SF réelle
f_1 = 1/2
for n in range(1, ordre_max):
    f_1 += ((-2 * ((-1)**n - 1)) / (n * np.pi)**2) * np.cos(n * np.pi * t)

#SF complexe
f_2 = 0
for n in range(-100,100):
    if (n!=0):
        f_2 += (-1/(n*np.pi)**2)*((-1)**n - 1) * np.exp(1j * np.pi * n * t)
    else:
        f_2 += 1/2



# Tracer le graphique
plt.plot(t, f_1, 'y', label = "SF réelle")
plt.plot(t, f_2.real, 'r', linestyle = '-.', label = "SF complexe")
plt.title('Série de Fourier de $f(t)$')
plt.xlabel('t')
plt.ylabel('$f(t)$')
plt.grid(True)
plt.legend()
plt.savefig('TP2_IMN359_3d.jpg')
plt.show()
