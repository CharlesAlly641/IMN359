import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Définition de la fonction f(t)
def Lambda(t):
    return 1 - np.abs(t) if np.abs(t) < 1 else 0

# Coefficients de la série de Fourier complexe
def cn(n):
    if n == 0:
        return 1/2
    return (-1 / (n * np.pi)**2) * ((-1)**n - 1)

# Approximation de la série de Fourier avec N harmoniques
def SF_approximation(t, N):
    resultat = 0
    for n in range(-N, N):
        resultat += cn(n) * np.exp(1j * n * np.pi * t)
    return resultat.real

# Calcul de l'erreur quadratique moyenne
def EQM(N):
    integrale = lambda t: (Lambda(t) - SF_approximation(t, N))**2
    erreur, _= quad(integrale, -1, 1)
    return erreur / 2

# Paramètres
t = np.linspace(-1, 1, 500)
harmoniques = [2, 20, 100]

# Graphique
plt.figure(figsize=(10, 6))
plt.plot(t, [Lambda(t) for t in t], label="Lambda(t)", color='black')

for N in harmoniques:
    approximation = [SF_approximation(t, N) for t in t]
    plt.plot(t, approximation, label=f"Approximation avec {N} harmoniques")

plt.legend()
plt.title("Approximation de la série de Fourier pour différents nombres d'harmoniques")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.grid(True)
plt.savefig('TP2_IMN359_3e.jpg')
plt.show()


# Affichage de l'erreur pour chaque approximation
for N in harmoniques:
    error = EQM(N)
    print(f"Erreur quadratique moyenne pour N = {N} : {error}")
