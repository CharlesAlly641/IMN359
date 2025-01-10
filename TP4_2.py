import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.io import loadmat
from dct2 import dct2
from idct2 import idct2
from scipy.fft import fft2, fftshift, ifftshift, ifft2
from perform_thresholding import perform_thresholding
import pandas as pd

lena = loadmat("lena.mat")['lena']
M_valeurs = [100,500,1000]

# Fonction pour calculer le SNR
def SNR(original, reconstructed):
    return 20 * np.log10(np.linalg.norm(original) / np.linalg.norm(original - reconstructed))

def erreur_approx(original, reconstructed):
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

# Question 2 a) : Approximation Fourier
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

Images_filtrees = []

# Fonction d'approximation en Fourier
def approximation_fourier(image, M_valeurs):
    fft_image = fftshift(fft2(image))
    Images_filtrees = []
    snr_valeurs = []
    erreurs = []

    for M in M_valeurs:
        fft_filtree = filtre_passe_bas(fft_image, M)
        image_reconstruite = np.abs(ifft2(ifftshift(fft_filtree)))
        Images_filtrees.append((M, image_reconstruite))

        snr = SNR(image, image_reconstruite)
        erreur = erreur_approx(image, image_reconstruite)
        snr_valeurs.append(snr)
        erreurs.append(erreur)

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_valeurs,
        'Erreur relative (linéaire)': erreurs
    })
    print("Tableau des résultats - Approximation Fourier :")
    print(df)

    # Affichage des résultats
    plt.figure(figsize=(18, 6))
    for i, (M, img) in enumerate(Images_filtrees):
        plt.subplot(1, len(M_valeurs), i + 1)
        plt.imshow(img)
        plt.title(f'M = {M}\nSNR = {snr_valeurs[i]:.2f} dB\nErreur relative = {erreurs[i]:.4f}')
        plt.axis('off')

    plt.savefig('2a_approximation_fourier.jpg')
    plt.show()

approximation_fourier(lena, M_valeurs)


# Question 2 b) : Approximation Cosinus
# DCT
def approximation_dct(image, M_values):
    snr_valeurs_dct = []
    erreurs_dct = []
    fig, axs = plt.subplots(1, len(M_values), figsize=(15, 5))

    for j, M in enumerate(M_values):
        dct_image = dct2(image)

        # Créer un masque pour garder les M coefficients de basse fréquence
        dct_mask = np.zeros_like(dct_image)
        dct_mask[:M, :M] = 1

        # On applique le mask à la DCT
        dct_image_approx = dct_image * dct_mask

        # On reconstruit l'image
        reconstructed_image = idct2(dct_image_approx)

        # Calculer le SNR et l'erreur relative
        snr = SNR(image, reconstructed_image)
        erreur = erreur_approx(image, reconstructed_image)
        snr_valeurs_dct.append(snr)
        erreurs_dct.append(erreur)

        # Afficher l'image reconstruite
        axs[j].imshow(reconstructed_image)
        axs[j].set_title(f'M = {M}\nSNR = {snr:.2f} dB\nErreur relative = {erreur:.4f}')
        axs[j].axis('off')
    plt.savefig('2b_approximation_cosinus.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_valeurs_dct,
        'Erreur relative (linéaire)': erreurs_dct
    })

    print("Tableau des résultats - Approximation DCT :")
    print(df)
    plt.show()

approximation_dct(lena, M_valeurs)


# DCT locale
def approximation_dct_locale(image, M_values, block_size):
    x, y = image.shape
    snr_valeurs = []
    erreur_valeurs = []

    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))

    for k, M in enumerate(M_values):
        locDCT_full = np.zeros_like(image)
        reconstruction = np.zeros_like(image)

        # Parcourir les blocs de l'image
        for i in range(int(x/block_size)):
            for j in range(int(y/block_size)):
                # Indices du bloc
                seli_full_r = np.array([i * block_size, (i + 1) * block_size])
                seli_full_c = np.array([j * block_size, (j + 1) * block_size])

                # Appliquer la DCT locale
                bloc_dct = dct2(lena[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]])

                # Créer un masque pour garder les M premiers coefficients
                dct_mask = np.zeros_like(bloc_dct)
                dct_mask[:M // block_size, :M // block_size] = 1

                # On applique le mask à la DCT locale
                locDCT_full[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]] = bloc_dct * dct_mask

                # Reconstruire l'image
                reconstruction[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]] = idct2(
                locDCT_full[seli_full_r[0]:seli_full_r[1], seli_full_c[0]:seli_full_c[1]])

        # Calculer le SNR et l'erreur relative
        snr = SNR(image, reconstruction)
        erreur = erreur_approx(image, reconstruction)
        snr_valeurs.append(snr)
        erreur_valeurs.append(erreur)

        # Afficher l'image reconstruite pour cette valeur de M
        axs[k].imshow(reconstruction)
        axs[k].set_title(f'M = {M}\nSNR = {snr:.2f} dB\nErreur relative = {erreur:.4f}')
        axs[k].axis('off')
    plt.savefig('2b_approximation_cosinus_locaux.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_valeurs,
        'Erreur relative (linéaire)': erreur_valeurs
    })
    print("Tableau des résultats - Approximation DCT Locale :")
    print(df)

    plt.show()

approximation_dct_locale(lena, M_valeurs, 32)


# Question 2 c) : Ondelettes de Harr
def approximation_ondelettes(image, M_values, Jmax, Jmin):
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

            # Mise à jour de l'image
            image_copy[0:2**j, 0:2**j] = temp2

        # Conserver uniquement les M coefficients les plus significatifs (approximation)
        image_approx = np.zeros_like(image_copy)
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
        snr = SNR(image, image_approx)  #
        snr_values.append(snr)

        # Calcul de l'erreur relative
        erreur = erreur_approx(image, image_approx)
        relative_errors.append(erreur)

        # Affichage de l'image approximée
        axs[m].imshow(image_approx)
        axs[m].set_title(f'M = {M}\nSNR = {snr:.2f} dB\nErreur relative = {erreur:.4f}')
        axs[m].axis('off')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': relative_errors
    })
    print("Tableau des résultats - Approximation Ondelettes :")
    print(df)

    plt.savefig('2c_approximation_ondelettes.jpg')
    plt.show()

approximation_ondelettes(lena, M_valeurs, 9, 1)


# Question 2 d) : Approximations non linéaires
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

        # Affichage de l'image reconstruite
        axs[i].imshow(image_filtered)
        axs[i].set_title(f'M={M}\nSNR={snr:.2f} dB\nErreur relative ={erreur:.4f}')
        axs[i].axis('off')
    plt.savefig('2d_approximation_fourier.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation Fourier Nonlineaire :")
    print(df)

    plt.show()

approximation_fourier_nonlineaire(lena, M_valeurs)


# Approximation DCT
def approximation_dct_nonlineaire(image, M_values):
    snr_values = []
    error_values = []
    images_approx = []

    fig, axs = plt.subplots(1, len(M_values), figsize=(18, 6))  # Créer les sous-figures

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

    plt.savefig('2d_approximation_cosinus.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation DCT Nonlineaire :")
    print(df)

    plt.show()

approximation_dct_nonlineaire(lena, M_valeurs)


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

    plt.savefig('2d_approximation_cosinus_locaux.jpg')

    df = pd.DataFrame({
        'M': M_valeurs,
        'SNR (dB)': snr_values,
        'Erreur relative (linéaire)': error_values
    })
    print("Tableau des résultats - Approximation DCT Locale Nonlineaire :")
    print(df)

    plt.show()

approximation_dct_locale_nonlineaire(lena, M_valeurs, 32)


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

        # On garde uniquement les M plus grands coefficients
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
        snr = SNR(image, image_approx)
        snr_values.append(snr)

        # Calcul de l'erreur relative
        erreur = erreur_approx(image, image_approx)
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

    plt.savefig('2d_approximation_ondelettes.jpg')
    plt.show()

approximation_ondelettes_nonlineaire(lena, M_valeurs, 9, 1)