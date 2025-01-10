import numpy as np

# Définissions les matrices représentant le système d'équations
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -3, 4]])
B = np.array([0, -1, 4])

# Résolution du système d'équations
solution = np.linalg.solve(A, B)

# Affichage de la solution
print("Solution :")
print(f"x = {solution[0]}")
print(f"y = {solution[1]}")
print(f"z = {solution[2]}")


