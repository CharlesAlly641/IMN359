import numpy as np
from scipy.io import loadmat


def dct(a, n=0):
    """
    DCT - discrete cosinus transform
    Return the discrete cosine transform of a.
    The vector b is the same size as b and contains the
    discrete cosine transform coefficients.

    With n > 0, the function pads or truncate the vector a
    to length n before transforming.

    If a is a matrix, the dct operation is applied to each
    column. This transform can be inverted using idct.

    Parameters
    ----------
    a : array on which the dct will be applied
    n : length of the truncation (default 0 means no truncation).
    Returns
    -------
    b : results of the dct

    References
    ----------
    1) A. K. Jain, "Fundamentals of Digital Image
       Processing", pp. 150-153.
    2) Wallace, "The JPEG Still Picture Compression Standard",
      Communications of the ACM, April 1991.
    https://www.tutorialspoint.com/execute_matlab_online.php
    """

    if a.size == 0:
        b = []
        return b

    if n == 0:
        n = a.shape[0]

    # If input is a vector, make it a column
    do_trans = False
    if len(a.shape) == 1:
        do_trans = True
        a = np.reshape(a, (a.shape[0], -1))

    m = a.shape[1]

    # Pad or truncate input if necessary
    if a.shape[0] < n:
        aa = np.zeros(n, m)
        aa[0:n] = a
    else:
        aa = a[0:n]

    # Compute weights to multiply DCT coefficients
    w = np.reshape(np.arange(0, n), (n, -1))
    ww = np.exp(-1j * w * np.pi/(2*n))/np.sqrt(2*n)
    ww[0] = ww[0] / np.sqrt(2)

    if n % 2 == 1 or isinstance(a, complex):  # odd case
        # Form intermediate even-symetric matrix
        y = np.zeros((2*n, m), dtype=complex)
        y[0:n] = aa
        y[n:2*n, :] = np.flipud(aa)

        # Compute the FFt and keep the appropriate portion
        yy = np.fft.fft(y, axis=0)
        yy = yy[0:n, :]

    else:  # even case
        # Re-order the elements of the columns of x
        y = np.zeros_like(aa)
        y[0:int(n/2)] = aa[::2]
        y[int(n/2):n] = np.flipud(aa)[::2]
        np.savetxt('text', y)

        yy = np.fft.fft(y, axis=0)
        ww = 2 * ww  # double the weights for even-length case

    # Multiply FFT by weights
    b = ww * np.ones((n, m)) * yy
    if ~isinstance(a, complex):
        b = np.real(b)
    
    if do_trans:
        b = b.T

    return b
