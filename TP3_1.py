import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.fft import fft, fftshift, ifft

#Gausienne 1D (#1a)
def gaussienne(t, sigma):
    return np.exp(-t**2 / (2*sigma**2))

sigma = 6
t= np.linspace(-512,512,1024)
gauss = gaussienne(t, sigma)
plt.plot(t,gauss)
plt.title('Gaussienne 1D')
plt.savefig('1a_gaussienne.jpg')
plt.show()

#TF de la gaussienne (#1a)
def gausienne_TF(f, sigma):
    return np.sqrt(2 * (sigma**2) * np.pi) * np.exp(-2 * (np.pi**2) * (f**2) * (sigma**2))

delta_t = t[1] - t[0]
delta_f = 1 / (1024 * delta_t)
f = np.linspace(-512 * delta_f, 512 * delta_f, 1024)
G = gausienne_TF(f, sigma)
plt.plot(f, G)
plt.title('Gaussienne TF')
plt.savefig('1a_gaussienne_tf.jpg')
plt.show()

#TF de la gaussienne avec fft (#1b)
G_numerique = np.abs(fftshift(fft(gauss)))
plt.plot(f, G_numerique)
plt.title('Gaussienne TF Numérique')
plt.savefig('1b_gaussienne_tf.jpg')
plt.show()

#Piece regular (#1c)
g = loadmat("piece-regular.mat")['x0']
g = g.squeeze()
t = np.linspace(0, 1000, np.shape(g)[0])
delta_tg = t[1] - t[0]
delat_fg = 1/(1024 * delta_t)
f = np.linspace(-512 * delta_f, 512 * delta_f, 1024)
g_bruit = g + 0.1 * np.random.rand(1024)
plt.subplot(2,1,1)
plt.plot(t,g)
plt.subplot(2,1,2)
plt.plot(t,g_bruit)
plt.savefig('1c_piece_regular.jpg')
plt.show()
G_bruit = fftshift(fft(g_bruit))
#Vérification qu'on a la bonne TF, on devrait réobtenir g_bruit
#iFT = ifft(fftshift(G_bruit))
#plt.plot(t,iFT)
#plt.show()

#Convolution (#1d)
g_conv = np.convolve(g_bruit, gauss, "same")
plt.plot(g_conv)
plt.title('Convolution du signal bruité et du filtre gaussien')
plt.savefig('1d_convolution.jpg')
plt.show()

produit = G_numerique * G_bruit
produitInverse = np.abs(ifft(fftshift(produit)))
plt.plot(t, produitInverse)
plt.title('ifft de la multiplication de G_numerique et G_bruit')
plt.savefig('1d_ifft.jpg')
plt.show()

#Théorème de Plancherel (#1div)
energie_temps = np.sum(np.abs(g_conv) ** 2)
energie_frequence = np.sum(np.abs(produit) ** 2) / len(produit)
erreur = (energie_frequence - energie_temps) / (energie_temps + energie_frequence)

print(f"Énergie dans l'espace du temps : {energie_temps}")
print(f"Énergie dans l'espace des fréquences : {energie_frequence}")
print(f"Erreur relative entre les deux énergies : {erreur : .3f}")


