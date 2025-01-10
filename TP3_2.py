import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft2, fftshift, ifft2

#Filtre cosinusoidal 1D (#2a)
x = np.linspace(-np.pi, np.pi, 512)

#On crée une fonction h qui, aux points clés de la fonction porte
#donne le même résultat. h(0) = 1 et h(+-pi) = 0
h = (1 + np.cos(x))/2
plt.plot(x, h)
plt.title("Filtre cosinusoidal 1D avec support [-π, π]")
plt.savefig('2a_filtre_cosinusoidal.jpg')
plt.show()

#Généralisation du filtre (#2b)
x = np.linspace(-np.pi, np.pi, 512)
y = np.linspace(-np.pi, np.pi, 512)
X, Y = np.meshgrid(x, y)

H = (1 + np.cos(X))/2 * (1 + np.cos(Y))/2

plt.contourf(X, Y, H, levels=100)
plt.colorbar(label="Amplitude")
plt.title("Filtre cosinusoidal 2D avec support [-π, π] x [-π, π]")
plt.savefig('2b_filtre_cosinusoidal_generalisation.jpg')
plt.show()

#fft2 de Lena.mat (#2c)
lena = loadmat("lena.mat")['M']

lena_fft = fftshift(fft2(lena))
Image_spectre = np.log(np.abs(lena_fft))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(lena)
plt.title('Lena')
plt.subplot(1, 2, 2)
plt.imshow(Image_spectre)
plt.title('Spectre de Fourier (échelle logarithmique)')
plt.savefig('2c_fft2_lena.jpg')
plt.show()

#Lena et filtre cosinusoidal (#2d)
x = np.linspace(-np.pi, np.pi, lena.shape[1])
y = np.linspace(-np.pi, np.pi, lena.shape[0])
X, Y = np.meshgrid(x, y)

lena_fenetree = lena * H
lena_fenetree_fft = fftshift(fft2(lena_fenetree))
Image_spectre_fenetree = np.log(np.abs(lena_fenetree_fft))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(lena_fenetree)
plt.title('Lena fenêtrée')
plt.subplot(1, 2, 2)
plt.imshow(Image_spectre_fenetree)
plt.title('Spectre de Fourier fenêtré (échelle logarithmique)')
plt.savefig('2d_lena_filtre_cosinusoidal.jpg')
plt.show()

#Passe-bas dans Fourier (#2e)
def filtre_passe_bas(fft_image, M):
    x, y = fft_image.shape
    centre_x = x // 2
    centre_y = y // 2
    moitie_cote = int(np.sqrt(M) // 2)

    #1 à l'intérieur de notre carré √M x √M, 0 à l'extérieur
    filtre = np.zeros_like(fft_image)
    filtre[centre_x - moitie_cote:centre_x + moitie_cote, centre_y - moitie_cote:centre_y + moitie_cote] = 1

    fft_filtree = fft_image * filtre

    return fft_filtree

M_valeurs = [100, 500, 1000]
Images_filtrees = []
Spectres_filtres = []

for M in M_valeurs:
    fft_filtree = filtre_passe_bas(lena_fft, M)
    lena_filtree = np.abs(ifft2(fftshift(fft_filtree)))
    Images_filtrees.append((M, lena_filtree))
    Spectres_filtres.append((M, fft_filtree))

plt.figure(figsize=(18, 6))

#Pour chaque valeur à l'indice i dans Spectres_filtres, on récupère M et fft_image
for i, (M, fft_image) in enumerate(Spectres_filtres):
    #On met + 1 pour éviter les valeurs complexes
    Image_spectre = np.log(np.abs(fft_image) + 1)
    plt.subplot(1, 3, i + 1)
    plt.imshow(Image_spectre)
    plt.title(f'Spectre filtré avec M={M}')
    plt.axis('off')
plt.savefig('2e_passe_bas_fourier.jpg')
plt.show()

#Regénération des images de Lena (#2f)

#Pour chaque valeur à l'indice i dans Images_filtrees, on récupère M et image
for i, (M, image) in enumerate(Images_filtrees):

    #On calcule l'ecart quadratique
    eqm = np.mean((lena - image) ** 2)

    #On divise le nombre de points M par le nombre de points total
    pourcentage_points = (M / (lena.shape[0] * lena.shape[1])) * 100

    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.title(f'M={M}\nEQM={eqm:.2f}\n% Points={pourcentage_points:.2f}%')
    plt.axis('off')

plt.savefig('2f_passe_bas_lena.jpg')
plt.show()







