import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft2, fftshift, ifft2, ifftshift

IRM = loadmat("IRM.mat")['IRM']

#Échantillonnage du signal IRM avec une ligne sur N en moins (#3a)
def Remplacement_une_ligne_sur_N(k_space, N):
    k_space = k_space.copy()

    #On parcourt les lignes de 0 à k_space.shape[0] en sautant N lignes
    for i in range(0, k_space.shape[0], N):

        #On parcourt toutes les lignes de k_space
        for j in range(0, k_space.shape[1]):

            #On assigne 0 à la coordonnée [i,j] de k_space
            k_space[i,j] = 0

    return k_space

def Image_reconstruite(k_space, N_valeurs):
    fig, axes = plt.subplots(1, len(N_valeurs), figsize=(15, 5))
    for i, N in enumerate(N_valeurs):
        k_space = Remplacement_une_ligne_sur_N(k_space, N)
        ifft_k_space = np.abs(ifft2(k_space))
        pourcentage_points = (1 - 1/N) * 100
        axes[i].imshow(ifft_k_space)
        axes[i].set_title(f'N={N}\n % Points={pourcentage_points:.2f}%')
    plt.savefig('3a_lignes.jpg')
    plt.show()

N_valeurs = [2, 3, 4]
Image_reconstruite(IRM, N_valeurs)

#Échantillonnage du signal IRM avec une colonne sur N en moins (#3b)
def Remplacement_une_colonne_sur_N(k_space, N):
    k_space = k_space.copy()

    #On parcourt les colonnes de 0 à k_space.shape[1] en sautant N colonnes
    for j in range(0, k_space.shape[1], N):

        #On parcourt toutes les lignes de k_space
        for i in range(0, k_space.shape[0]):

            #On assigne 0 à la coordonnée [i,j] de k_space
            k_space[i,j] = 0

    return k_space

def Image_reconstruite(k_space, N_valeurs):
    fig, axes = plt.subplots(1, len(N_valeurs), figsize=(15, 5))
    for i, N in enumerate(N_valeurs):
        k_space = Remplacement_une_colonne_sur_N(k_space, N)
        ifft_k_space = np.abs(ifft2(k_space))
        pourcentage_points = (1 - 1/N) * 100
        axes[i].imshow(ifft_k_space)
        axes[i].set_title(f'N={N}\n % Points={pourcentage_points:.2f}%')
    plt.savefig('3b_colonnes.jpg')
    plt.show()

Image_reconstruite(IRM, N_valeurs)

#Zero padding (#3c)
def zero_padding(k_space, taille):
    padded_k_space = np.zeros(taille, k_space.dtype)

    #On parcourt toutes les lignes de k_space
    for i in range(0, k_space.shape[0]):

        #On parcourt toutes les colonnes de k_space
        for j in range(0, k_space.shape[1]):

            #On met les valeurs se trouvant à l'indice [i,j] de k_space
            #dans le nouveau k_space de zero padding
            padded_k_space[i,j] = k_space[i,j]

    return padded_k_space

def Image_reconstruite(k_space, taille):
    fig, axes = plt.subplots(1, len(taille), figsize=(15, 5))
    for i, taille in enumerate(taille):
        padded_k_space = zero_padding(k_space, (taille, taille))
        ifft_padded_k_space = np.abs(ifft2(padded_k_space))
        axes[i].imshow(ifft_padded_k_space)
        axes[i].set_title(f'Taille={taille}x{taille}')
    plt.savefig('3c_zero_padding.jpg')
    plt.show()

taille = [600, 850, 1024]

Image_reconstruite(IRM, taille)

#Échantillonnage radial (#3d)
def echantillonnage_radial(k_space, theta, tolerance):
    mask = np.zeros(k_space.shape)

    #On génère des angles de 0 à pi espacés de theta
    angles = np.arange(0, np.pi, theta)

    #On crée une grille 2D avec les coordonnées des couples x,y
    x = np.arange(-k_space.shape[0] // 2, k_space.shape[0] // 2)
    y = np.arange(-k_space.shape[1] // 2, k_space.shape[1] // 2)
    X, Y = np.meshgrid(x, y)
    phi = np.arctan2(Y, X)

    #On évalue si le point X,Y est proche du rayon qu'on veut traçer.
    #S'il est proche, alors on lui associe la valeur 1.
    for angle in angles:
        difference = np.abs(phi - angle)
        rayon = difference < tolerance
        mask[rayon] = 1

    return mask*k_space

def Image_reconstruite(k_space, theta_valeurs):
    fig, axes = plt.subplots(1, len(theta_valeurs), figsize=(18, 5))
    for i, theta in enumerate(theta_valeurs):
        theta = np.radians(theta)
        k_space_echantillonne = echantillonnage_radial(k_space, theta, np.pi/180)
        ifft_k_space_echantillonne= np.abs(ifft2(k_space_echantillonne))

        #Pour le pourcentage de points, on prend seulement les valeurs qui ne sont pas à 0
        #dans le k_space_echantillonne.
        pourcentage_points = np.count_nonzero(k_space_echantillonne) / k_space.size * 100
        axes[i].imshow(ifft_k_space_echantillonne)
        axes[i].set_title(f'θ={theta:.2f} rad\n % Points={pourcentage_points:.2f}%')
        plt.savefig('3d_echantillonnage_radial.jpg')
    plt.show()

theta_values = [5, 30, 180]
Image_reconstruite(IRM, theta_values)


#Échantillonnage aléatoire (#3e)
def echantillonnage_aleatoire(k_space, P):
    k_space_echantillonne = np.zeros_like(k_space)
    nombre_points_total = k_space.size
    nombre_points = int(P *nombre_points_total)

    #On choisit aleatoirement les indices de nombre_points parmi nombre_points_total
    indices = np.random.choice(nombre_points_total, nombre_points)

    index = 0

    #On parcourt toutes les lignes de k_space
    for i in range(k_space.shape[0]):

        #On parcourt toutes les colonnes de k_space
        for j in range(k_space.shape[1]):

            #Si l'index se retrouve dans les indices retenus après
            #l'échantillonnage alors on l'inclut dans notre nouveau
            #k_space_echantillonne
            if index in indices:
                k_space_echantillonne[i,j] = k_space[i,j]
            index += 1

    return k_space_echantillonne

def Image_reconstruite(k_space, P_valeurs):
    fig, axes = plt.subplots(1, len(P_valeurs), figsize=(15, 5))
    for i, P in enumerate(P_valeurs):
        k_space_echantillonne = echantillonnage_aleatoire(k_space, P)
        ifft_k_space_echantillonne = np.abs(ifft2(k_space_echantillonne))
        axes[i].imshow(ifft_k_space_echantillonne)
        axes[i].set_title(f'% Points={P*100:.2f}%')
        axes[i].axis('off')
    plt.savefig('3e_echantillonnage_aleatoire.jpg')
    plt.show()

P_valeurs = [0.1, 0.5, 0.8]

Image_reconstruite(IRM, P_valeurs)

