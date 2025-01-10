import matplotlib.pyplot as plt
from TP1_2b import serie_geometrique

# Affectation de la valeur x = 0.5
x = 0.5
valeur_attendu = 1 / (1 - x)

# Nombre maximum de termes à considérer dans l'approximation
max_termes = 10

# Calcul de l'erreur pour chaque nombre de termes
erreurs = []
for n in range(max_termes):
    somme_partielle = serie_geometrique(x, n)
    erreur = abs(valeur_attendu - somme_partielle)
    erreurs.append(erreur)

# Graphique illustrant  l'erreur en fonction du nombre de termes à considérer
plt.plot(range(max_termes), erreurs)
plt.xlabel("Nombre de termes")
plt.ylabel("Erreur absolue")
plt.title("Erreur de la série géométrique tronquée pour x = 0.5")
plt.grid(True)
plt.xlim(0)
plt.ylim(0)
plt.show()