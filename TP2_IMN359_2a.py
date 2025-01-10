import matplotlib.pyplot as plt
import numpy as np

#Définir l'intervalle de temps
t = np.arange(0, 3*np.pi,0.01)

#Définir la fonction y
y1 = np.sin(t)**2

#Représentation de la période sur le graphique
plt.axvline(x=np.pi, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=2*np.pi, color='r', linestyle='--')
plt.axvline(x=3*np.pi, color='r', linestyle='--')

#Caractéristiques du graphique
plt.plot(t,y1)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Période fondamentale de $\sin^2(t)$")
plt.show()