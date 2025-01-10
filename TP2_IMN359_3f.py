import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-10,10,10000)
f = 0

#Approximation de f(t) par la SF complexe
for n in range(-10000,10000):
    if (n!=0):
        f += (1/(np.pi * n)) * np.sin(n*np.pi/2)* np.exp(1j * n * t * np.pi/10)
    else:
        f += 1/2


# Tracer le graphique
plt.plot(t, f, 'b')
plt.title('Série de Fourier de $f(t)$')
plt.xlabel('t')
plt.ylabel('$f(t)$')
plt.grid(True)
plt.savefig('TP2_IMN359_f_SF.png')
plt.show()
