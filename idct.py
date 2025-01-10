import numpy as np


def idct(b, n=0):
    """
    IDCT Inverse discrete cosine transform.
    Return the inverse discrete cosine transform of b.
    The vector b is the same size as b and contains the
    discrete cosine transform coefficients.

    With n > 0, the function pads or truncate the vector a
    to length n before transforming.

    If b is a matrix, the idct operation is applied to each
    column. This transform can be inverted using dct.

    Parameters
    ----------
    b : array on which the dct will be applied
    n : length of the truncation (default 0 means no truncation).
    Returns
    -------
    a : results of the dct

    References
    ----------
    1) A. K. Jain, "Fundamentals of Digital Image
       Processing", pp. 150-153.
    2) Wallace, "The JPEG Still Picture Compression Standard",
      Communications of the ACM, April 1991.
    https://www.tutorialspoint.com/execute_matlab_online.php
    """

    if b.size == 0:
        a = []
        return a

    if n == 0:
        n = b.shape[0]

    # If input is a vector, make it a column
    do_trans = False
    if len(b.shape) == 1:
        do_trans = True
        b = np.reshape(b, (b.shape[0], -1))

    m = b.shape[1]

    # Pad or truncate input if necessary
    if b.shape[0] < n:
        bb = np.zeros(n, m)
        bb[0:n] = b
    else:
        bb = b[0:n]

    # Compute weights to multiply DCT coefficients
    w = np.reshape(np.arange(0, n), (n, -1))
    ww = np.sqrt(2*n) * np.exp(1j * w * np.pi/(2*n))
    ww[0] = ww[0] / np.sqrt(2)

    if n % 2 == 1 or isinstance(b, complex):  # odd case
        # Form intermediate even-symetric matrix
        W = ww * np.ones((n, m))
        yy = np.zeros((2*n, m), dtype=complex)
        yy[0:n] = ww * bb 
        yy[n+1:2*n, :] = -1j * W[1:n] * np.flipud(bb[1:n])

        # Extract inverse DCT
        y = np.fft.ifft(yy, axis=0)
        a = y[0:n]

    else:  # even case
        # Re-order the elements of the columns of x
        W = ww * np.ones((n, m))
        yy = W * bb
        y = np.fft.ifft(yy, axis=0)

        # Re-order elements of each column according to equations (5.93) and 
        # (5.94) in Jain
        a = np.zeros_like(bb, dtype=complex)
        a[0::2] = y[0:int(n/2)]
        a[1::2] = np.flipud(y)[0:int(n/2)]

    if ~isinstance(b, complex):
        a = np.real(a)

    if do_trans:
        a = a.T

    return a