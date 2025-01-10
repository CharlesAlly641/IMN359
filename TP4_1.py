import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from dct import dct
from idct import idct
from perform_thresholding import perform_thresholding

#Illustration du signal piece-regular avec 1024 points (#1a)
x0 = loadmat('piece-regular.mat')['x0']
n = x0.shape[0]
t = np.linspace(0,1,n)
plt.plot(t,x0)
plt.title('1a_Signal 1D piece-regular')
#plt.savefig('piece-regular.png')
plt.show()


############################################################
# DCT locale
############################################################
# Approximation avec des cosinus locaux
#
# Ameliore l'approximation globale de la DCT
# On decoupe le signal en segments locaux  dans lesquels on
# fait la DCT JPEG est basée sur cette approche
# ###########################################################

coeff = 256
dct_full = dct(x0)
dct_linear = np.zeros((n, 1))
dct_linear[0:coeff] = dct_full[0:coeff]
f_dct_full = idct(dct_full)  # la full DCT reconstruction sans couper des coefficients
f_dct_full_lin = idct(dct_linear)  # approximation linear du piece-regular prenant les 128 premiers coefficients
f_dct_full_nonlin = idct(perform_thresholding(dct_full, coeff,
                                              'largest'))  # approximation non-linear prenant que les 128 plus grands coeffcients

# taille du segment
w = 32

# coefficients de la DCT locale
locDCT_full = np.zeros((n, 1))

# DCT locale
for i in range(int(n / w)):
    # indices du segment i
    seli_full = np.array([i * w, i * w + w])
    # DCT locale dans le segment i
    locDCT_full[seli_full[0]:seli_full[1]] = dct(x0[seli_full[0]:seli_full[1]])

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.abs(dct_full))
axs[1].plot(np.abs(locDCT_full))
plt.savefig('1b_taille_segment')
plt.show()

# On a 32 segments de taille 32 (1024 / 32 = 32). Donc, si on garde 128 coefficients au total,
# on garde que s coefficients (128/32=4) par segment.

# iDCT locale. On initialise tout à 0
local_segment = np.zeros((n, 1))
f_locDCT_full = np.zeros((n, 1))
f_locDCT_lin = np.zeros((n, 1))
f_locDCT_nonlin = np.zeros((n, 1))

s = int(coeff / (n / w))
for i in range(int(n / w)):
    # tous indices du segment i
    seli_full = np.array([i * w, i * w + w])
    # que les s premiers indices du segment i
    seli_cut = np.array([i * w, i * w + s])

    # la full DCT local prends tous les coefficients s a chaque segment
    f_locDCT_full[seli_full[0]:seli_full[1]] = idct(dct(x0[seli_full[0]:seli_full[1]]))

    # l'approximation lineaire garde que les s premiers coefficients
    # le reste des coefficients restent à zéro
    local_segment[seli_cut[0]:seli_cut[1]] = locDCT_full[seli_cut[0]:seli_cut[1]]
    f_locDCT_lin[seli_full[0]:seli_full[1]] = idct(local_segment[seli_full[0]:seli_full[1]])

    # l'approximation non-lineaire garde que les s plus grands coefficients dans chaque segment i
    # le reste des coefficients restent à 0
    f_locDCT_nonlin[seli_full[0]:seli_full[1]] = idct(
        perform_thresholding(locDCT_full[seli_full[0]:seli_full[1]], s, 'largest'))

f = x0
#print('Erreur quadratique moyenne avec une full Discrete Cosine transform (DCT) = ',
      #np.linalg.norm(f - f_dct_full) / np.linalg.norm(f))
#print('Erreur quadratique moyenne avec une full locale Discrete Cosine transform (locDCT) = ',
      #np.linalg.norm(f - f_locDCT_full) / np.linalg.norm(f))
#print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients lineair DCT coeffs = ',
      #np.linalg.norm(f - f_dct_full_lin) / np.linalg.norm(f))
#print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients lineair locDCT coeffs = ',
      #np.linalg.norm(f - f_locDCT_lin) / np.linalg.norm(f))
#print('\n')
#print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients non-linear DCT coeffs = ',
      #np.linalg.norm(f - f_dct_full_nonlin) / np.linalg.norm(f))
#print('Erreur quadratique moyenne avec ' + str(coeff) + ' coefficients non-linear local DCT coeffs = ',
      #np.linalg.norm(f - f_locDCT_nonlin) / np.linalg.norm(f))

fig, axs = plt.subplots(5, 1)
fig.tight_layout(pad=1.0)
axs[0].plot(f_locDCT_full)
axs[0].set_title('\n Reconstruction avec une full locDCT (tous les coefficients)')
axs[1].plot(f_dct_full_lin)
axs[1].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f - f_dct_full) / np.linalg.norm(f)))
axs[2].plot(f_dct_full_nonlin)
axs[2].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f - f_dct_full_nonlin) / np.linalg.norm(f)))
axs[3].plot(f_locDCT_lin)
axs[3].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f - f_locDCT_lin) / np.linalg.norm(f)))
axs[4].plot(f_locDCT_nonlin)
axs[4].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f - f_locDCT_nonlin) / np.linalg.norm(f)))
plt.savefig('1c_approximation_lineaire_nonlineaire')
plt.show()

# regardons de plus pres les coefficients coupés
local_lin = np.zeros((n, 1))
local_nonlin = np.zeros((n, 1))
for i in range(int(n / w)):
    # indices du segment i
    seli_full = np.array([i * w, i * w + w])
    # DCT locale dans le segment i
    locDCT_full[seli_full[0]:seli_full[1]] = dct(x0[seli_full[0]:seli_full[1]])

    # que les s premiers indices du segment i
    seli_cut = np.array([i * w, i * w + s])
    local_lin[seli_cut[0]:seli_cut[1]] = locDCT_full[seli_cut[0]:seli_cut[1]]

    # que les s plus grands coefficients du segment i
    local_nonlin[seli_full[0]:seli_full[1]] = perform_thresholding(locDCT_full[seli_full[0]:seli_full[1]], s, 'largest')

global_nonlin = perform_thresholding(locDCT_full, coeff, 'largest')

fig, axs = plt.subplots(6, 1)
axs[0].plot(np.abs(f))
axs[1].plot(np.abs(dct_full))
axs[2].plot(np.abs(locDCT_full))
axs[3].plot(np.abs(local_lin))
axs[4].plot(np.abs(local_nonlin))
axs[5].plot(np.abs(global_nonlin))
plt.show()

f_global_nonlin = np.zeros((n, 1))
for i in range(int(n / w)):
    seli_full = np.array([i * w, i * w + w])
    f_global_nonlin[seli_full[0]:seli_full[1]] = idct(global_nonlin[seli_full[0]:seli_full[1]])

#print('Erreur quadratique moyenne avec ' + str(coeff)
      #+ ' coefficients non-lineares pris sur la globalité de la local DCT coeffs = ',
      #np.linalg.norm(f - f_global_nonlin) / np.linalg.norm(f))

fig, axs = plt.subplots(6, 1)
fig.tight_layout(pad=1.0)
axs[0].plot(f_locDCT_full)
axs[0].set_title('\n Reconstruction avec une full locDCT (tous les coefficients)')
axs[1].plot(f_dct_full_lin)
axs[1].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f - f_dct_full) / np.linalg.norm(f)))
axs[2].plot(f_dct_full_nonlin)
axs[2].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineaires de la full DCT. Erreur: '
                 + str(np.linalg.norm(f - f_dct_full_nonlin) / np.linalg.norm(f)))
axs[3].plot(f_locDCT_lin)
axs[3].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f - f_locDCT_lin) / np.linalg.norm(f)))
axs[4].plot(f_locDCT_nonlin)
axs[4].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares d\'une DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f - f_locDCT_nonlin) / np.linalg.norm(f)))
axs[5].plot(f_global_nonlin)
axs[5].set_title('\n Approximation avec ' + str(coeff)
                 + ' coefficients non-lineares sur la globalite de la DCT locale avec des segments de taille ' + str(w)
                 + '. Erreur: ' + str(np.linalg.norm(f - f_global_nonlin) / np.linalg.norm(f)))

plt.show()


# Calculer toutes les erreurs déjà faites dans le code
errors = {
    "Méthode": [
        "DCT globale",
        "DCT locale",
        f"{coeff} Coefficients Linéaires (DCT globale)",
        f"{coeff} Coefficients Linéaires (DCT locale)",
        f"{coeff} Coefficients Non-linéaires (DCT globale)",
        f"{coeff} Coefficients Non-linéaires (DCT locale)",
        f"{coeff} Coefficients Non-linéaires Globaux (DCT locale globale)"
    ],
    "EQM": [
        np.linalg.norm(f - f_dct_full) / np.linalg.norm(f),
        np.linalg.norm(f - f_locDCT_full) / np.linalg.norm(f),
        np.linalg.norm(f - f_dct_full_lin) / np.linalg.norm(f),
        np.linalg.norm(f - f_locDCT_lin) / np.linalg.norm(f),
        np.linalg.norm(f - f_dct_full_nonlin) / np.linalg.norm(f),
        np.linalg.norm(f - f_locDCT_nonlin) / np.linalg.norm(f),
        np.linalg.norm(f - f_global_nonlin) / np.linalg.norm(f)
    ]
}

# Convertir en DataFrame pour un affichage clair
df_errors = pd.DataFrame(errors)

# Afficher le tableau
print(df_errors)
