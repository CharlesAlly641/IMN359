def serie_geometrique(x, ordre):
    #Calcul de la sommation
    somme = 0
    for n in range(ordre):
        somme += x ** n
    return somme

# Tests pour trois valeurs de x et un ordre de notre choix
x_valeurs = [0.1, 0.2, 0.5]  # valeurs de x
ordre = 10  # nombre de termes dans la série

# Affichage des résultats obtenus
for x in x_valeurs:
    print(f"Série ayant la valeur x = {x} (approximée avec {ordre} termes) : {serie_geometrique(x, ordre)}")