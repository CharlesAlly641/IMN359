import numpy as np
import matplotlib.pyplot as plt

def porte_modifiee(t):
    # La fonction est 1 pour |t| < 5 ou |t| > 15, et 0 pour 5 < |t| < 15
    return np.where(np.abs(t) < 5, 1, np.where(np.abs(t) > 15, 1, 0))

# Exemple d'utilisation
t = np.linspace(-10, 10, 10000)
y = porte_modifiee(t)

# Affichage de la fonction
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Fonction f(t)')
plt.grid(True)
plt.savefig('TP2_IMN359_f_ft.png')
plt.show()
