import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.io import loadmat
from dct import dct
from idct import idct
from dct2 import dct2
from idct2 import idct2
from snr import snr
from scipy.fft import fft2, fftshift, ifftshift, ifft2, rfft
from perform_thresholding import perform_thresholding
import pandas as pd

# Le code se retrouvant dans ce fichier est le même que celui dans le fichier TP4_2,
# mais, pour intégrer les images dans le latex, j'ai changé le nom dans plt.savefig!

phantom = loadmat("phantom.mat")['phantom']
M_valeurs = [100, 500, 1000]

# Fonction pour calculer le SNR
def SNR(original, reconstructed):
    return 20 * np.log10(np.linalg.norm(original) / np.linalg.norm(original - reconstructed))

def erreur_approx(original, reconstructed):
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

# Approximation de Fourier
def approximation_fourier_nonlineaire(image, M_values):
    # Calcul de la FFT 2D de l'image
    image_fft = fftshift(fft2(image))
    snr_values = []
    error_values = []
    images_filtered = []

    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))

    for i, M in enumerate(M_values):
        # On garde les M plus grands coefficients
        fft_filtered = perform_thresholding(image_fft, M, 'largest')
        image_filtered = np.abs(ifft2(ifftshift(fft_filtered)))

        images_filtered.append(image_filtered)
        snr = SNR(image, image_filtered)
        erreur = erreur_approx(image, image_filtered)
        snr_values.append(snr)
        error_values.append(erreur)

        axs[i].imshow(image_filtered)
        axs[i].set_title(f'M={M}\nSNR={snr:.2f} dB\nErreur relative ={erreur:.4f}')
        axs[i].axis('off')
    plt.savefig('3_phantom_fourier.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation Fourier Nonlineaire :")
    print(df)

    plt.show()

approximation_fourier_nonlineaire(phantom, M_valeurs)


# Approximation DCT
def approximation_dct_nonlineaire(image, M_values):
    snr_values = []
    error_values = []
    images_approx = []

    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))

    for i, M in enumerate(M_values):
        dct_image = dct2(image)

        # On garde les M plus grands coefficients
        dct_thresholded = perform_thresholding(dct_image, M, 'largest')

        # Reconstruction de l'image
        reconstructed_image = idct2(dct_thresholded)

        # Calcul du SNR et de l'erreur relative
        snr = SNR(image, reconstructed_image)
        error = erreur_approx(image, reconstructed_image)

        images_approx.append((M, reconstructed_image, dct_thresholded))
        snr_values.append(snr)
        error_values.append(error)

        axs[i].imshow(reconstructed_image)
        axs[i].set_title(f'M = {M}\nSNR = {snr:.2f} dB\nErreur relative = {error:.4f}')
        axs[i].axis('off')

    plt.savefig('3_phantom_cosinus.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation DCT Nonlineaire :")
    print(df)

    plt.show()

approximation_dct_nonlineaire(phantom, M_valeurs)


# Approximation Cosinus Locaux
def approximation_dct_locale_nonlineaire(image, M_values, block_size):
    x, y = image.shape
    w = block_size

    snr_values = []
    error_values = []
    images_approx = []

    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))

    for idx, M in enumerate(M_values):
        locDCT_full = np.zeros_like(image)
        reconstruction = np.zeros_like(image)

        # Parcourir les blocs
        for i in range(int(x / w)):
            for j in range(int(y / w)):
                seli_full_r = np.array([i * w, (i + 1) * w])
                seli_full_c = np.array([j * w, (j + 1) * w])

                # Appliquer la DCT locale
                bloc_dct = dct2(image[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]])

                # On garde les M plus grands coefficients
                locDCT_full[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]] = perform_thresholding(
                    bloc_dct, M, 'largest')

                # Reconstruction du bloc
                reconstruction[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]] = idct2(
                    locDCT_full[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]]
                )

        # Calculer SNR et erreur relative
        snr_value = SNR(image, reconstruction)
        erreur = erreur_approx(image, reconstruction)

        snr_values.append(snr_value)
        error_values.append(erreur)
        images_approx.append((M, reconstruction))

        axs[idx].imshow(reconstruction)
        axs[idx].set_title(f'M = {M}\nSNR = {snr_value:.2f} dB\nErreur relative = {erreur:.4f}')
        axs[idx].axis('off')

    plt.savefig('3_phantom_cosinus_locaux.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation DCT Locale Nonlineaire :")
    print(df)

    plt.show()

approximation_dct_locale_nonlineaire(phantom, M_valeurs, 32)


# Approximation ondelettes
def approximation_ondelettes_nonlineaire(image, M_values, Jmax, Jmin):
    snr_values = []
    relative_errors = []
    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))
    for m, M in enumerate(M_values):
        image_copy = image.copy()

        for j in range(Jmax, Jmin - 1, -1):
            # Calcul des coefficients x et y
            x = (image_copy[0:2**j:2, 0:2**j] + image_copy[1:2**j:2, 0:2**j]) / np.sqrt(2)
            y = (image_copy[0:2**j:2, 0:2**j] - image_copy[1:2**j:2, 0:2**j]) / np.sqrt(2)
            temp = np.concatenate((x, y), axis=0)

            # Calcul des coefficients a et b
            a = (temp[0:2**j, 0:2**j:2] + temp[0:2**j, 1:2**j:2]) / np.sqrt(2)
            b = (temp[0:2**j, 0:2**j:2] - temp[0:2**j, 1:2**j:2]) / np.sqrt(2)
            temp2 = np.concatenate((a, b), axis=1)

            image_copy[0:2**j, 0:2**j] = temp2

        # On garde les M plus grands coefficients
        image_approx = np.zeros_like(image_copy)
        image_copy[:M, :M] = perform_thresholding(image_copy[:M, :M], M, 'largest')
        image_approx[:M, :M] = image_copy[:M, :M]

        # Reconstruction de l'image
        for j in range(Jmin, Jmax + 1):
            # Inverse des coefficients a et b
            temp2 = image_approx[0:2**j, 0:2**j]
            a = temp2[:2**j, :2**(j - 1)]
            b = temp2[:2**j, 2**(j - 1):2**j]
            temp = np.zeros_like(temp2)
            temp[:2**j, 0:2**j:2] = (a + b) / np.sqrt(2)
            temp[:2**j, 1:2**j:2] = (a - b) / np.sqrt(2)

            # Inverse des coefficients x et y
            x = temp[:2**(j - 1), :2**j]
            y = temp[2**(j - 1):2**j, :2**j]
            image_approx[:2**j, :2**j] = np.zeros_like(temp)
            image_approx[0:2**j:2, :2**j] = (x + y) / np.sqrt(2)
            image_approx[1:2**j:2, :2**j] = (x - y) / np.sqrt(2)

        # Calcul du SNR
        snr = SNR(image, image_approx)  # Utilisation de 'image' comme image originale
        snr_values.append(snr)

        # Calcul de l'erreur relative
        erreur = erreur_approx(image, image_approx)  # Même chose ici
        relative_errors.append(erreur)

        axs[m].imshow(image_approx)
        axs[m].set_title(f'M = {M}\nSNR = {snr:.2f} dB\nErreur relative = {erreur:.4f}')
        axs[m].axis('off')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': relative_errors
    })
    print("Tableau des résultats - Approximation Ondelettes Nonlineaire :")
    print(df)

    plt.savefig('3_phantom_ondelettes.jpg')
    plt.show()

approximation_ondelettes_nonlineaire(phantom, M_valeurs, 9, 1)