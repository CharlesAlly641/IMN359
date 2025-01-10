import numpy as np
from idct import idct

def idct2(a, mrows=0, ncols=0):
  """
  iDCT2 Compute 2-D inverse discrete cosine transform.
  Returns the discrete cosine transform of A.
  The matrix B is the same size as A and contains the
  discrete cosine transform coefficients.

  B = IDCT2(A,M,N) pads the matrix A with
  zeros to size M-by-N before transforming. If M or N is
  smaller than the corresponding dimension of A, iDCT2 truncates
  A. 

  This transform can be inverted using DCT2.

  Parameters
  ----------
  A : array on which the dct will be applied
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
  m = a.shape[0]
  n = a.shape[0]

  if (mrows == 0) and (ncols == 0):
      if (m > 1) and (n > 1): 
        b = idct(idct(a).T).T
        return b
      else:
        mrows = m 
        ncols = n

  # Padding for vector input.
  mpad = mrows
  npad = ncols
  if (m == 1) and (mpad > m):
    a[1, 0] = 0 
    m = 2
  if (n == 1) and (npad > n):
    a[0, 1] = 0 
    n = 2
  if m == 1:
    mpad = npad
    npad = 1

  # Transform 
  b = idct(a, mpad)
  if (m > 1) and (n > 1):
    b = idct(b.T, npad).T

  return b