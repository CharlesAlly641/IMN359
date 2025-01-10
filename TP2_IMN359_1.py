import numpy as np

#Définition de l'intégrale avec la méthode des rectangles
def integrale(fonction, a, b, n):
    """
       Paramètres:
       - fonction: la fonction à intégrer
       - a: borne inférieure de l'intervalle
       - b: borne supérieure de l'intervalle
       - n: nombre de sous-intervalles (rectangles)
       """
    largeur = (b - a) / n
    somme = 0

    for i in range(n):
        xi = a + i * largeur
        somme += fonction(xi) * largeur

    return somme

#Fonction à intégrer
def f(x):
    return np.cos(x) * np.cos(4 * x)


a = -np.pi  # borne inférieure
b = np.pi  # borne supérieure (par exemple de 0 à pi)
n = 1000  # nombre de rectangles

resultat = integrale(f, a, b, n)
print(f"L'approximation de l'intégrale de cos(x) * cos(4x) est: {resultat:.10f}")



