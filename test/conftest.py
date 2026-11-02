import os

os.environ.setdefault('OMP_NUM_THREADS', '1')          # tiny matrices: threading only slows NumPy
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
