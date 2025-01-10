import matplotlib.pyplot as plt
import numpy as np

#Définir l'intervalle de temps
t = np.arange(-100, 100, 0.01)

#Définir les fonctions
y1 = 2 * np.cos(t)
y2 = np.cos(t / 3)
y3 = 3 * np.cos(t / 5)
ys = 2 * np.cos(t) + np.cos(t / 3) + 3 * np.cos(t / 5)

#Définir la période sur la graphique
plt.axvline(x=30*np.pi, color='orange', linestyle='--')
plt.axvline(x=0, color='orange', linestyle='--')
plt.axvline(x=-30*np.pi, color='orange', linestyle='--')

#Caractéristiques du graphique
plt.plot(t, y1, 'g', t, y2, 'r', t, y3, 'b', t, ys, 'c', linewidth=2)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.legend({'2cos(t)','cos(t/3)','3cos(t/5)', '2cos(t) + cos(t/3) + 3cos(t/5)'})
plt.title('My cosines')
plt.show()