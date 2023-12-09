import os
import torch
import numpy as np
import pandas as pd
from fitting import preds_targets_zn, get_range_dat, polyfit3d, apply_fourier_2d, plot_rms_fit, linreg_PCA
from nuclr.train import Trainer
from nuclr.data import get_nuclear_data
from nuclr.data import semi_empirical_mass_formula, WS4_mass_formula, BW_mass_formula
import random
from math import factorial
from scipy.optimize import minimize
from numpy.polynomial.polynomial import polyval2d
from matplotlib import pyplot as plt



def radius_formula(data):
    N = data[["n"]].values
    Z = data[["z"]].values
    A = Z+N
    r0 = 1.2
    fake_rad = r0*A**(1/3)  # fm
    return fake_rad*(Z < 110)


def volume_term(Z, N):
    # Binding energy per nucleon is approximately constant due to saturation of nuclear force. Total binding scales with number of nucleons.
    A = Z + N
    aV = 16.58
    return aV * A

def surface_term(Z, N):
    # As nucleus gets bigger, surface area to volume ratio decreases. Surface tension lowers binding energy. 
    A = Z + N
    aS = -26.95
    return aS * A**(2/3)

def coulomb_term(Z, N):
    # Repulsive electrostatic energy between protons. Falls with A as protons spaced farther apart in larger nucleus.
    A = Z + N
    aC = -0.774
    return aC * Z**2 / (A**(1/3))

def asymmetry_term(Z, N):
    # Neutrons and protons have slightly different energies due to their different masses and charges. Deviations from N=Z raise the energy (lower binding). 
    A = Z + N
    I = N - Z
    aA = -31.51
    return aA * I**2 / A

def curvature_term(Z, N):
    # Nuclei not perfect spheres. Corrects for nuclear surface curvature effects beyond simple A^(2/3) term.
    A = Z + N
    aR = 14.77
    return aR * A**(1/3)

def coulomb_exchange(Z, N):
    # Quantum effect. Antisymmetrization of proton wavefunctions lowers direct Coulomb repulsion. Important for heavy nuclei.
    A = Z + N
    axC = 2.22
    return axC * Z**(4/3) / A**(1/3)

def wigner_term(Z, N):
    #  There is I (I + 1) or I (I + 4) form of isospin dependence arising in the seniority or supermultiplet models of ground-state energies. 
    A = Z + N
    I = N - Z
    aW = -43.4
    return aW * np.abs(I) / A

def surface_asym(Z, N):
    # Symmetry term should also correct surface energy. Ensures proper sign of Wigner term. Large effect on symmetry coefficient.
    A = Z + N
    I = N - Z
    ast = 55.62
    return ast * I**2 / A**(4/3)

def delta(Z, N):
    # Nucleons with the same quantum numbers can pair up, analogous to Cooper pairs in superconductors. This paired state has extra binding energy compared to unpaired nucleons.
    A = Z+N
    aP = 9.87
    delta0 = aP/A**(1/2)
    for i in range(len(A)):
        if ((N % 2 == 0) & (Z % 2 == 0))[i]:
            pass
        elif ((N % 2 == 1) & (Z % 2 == 1))[i]:
            delta0[i] = -delta0[i]
        else:
            delta0[i] = 0
    return delta0

def shell(Z, N):
    # Filling of quantized single-particle states leads to extra stabilization at shell closures. Accounts for "magic number" gaps.
    # warnings.filterwarnings("ignore", category=RuntimeWarning) 
    alpham = -1.9
    betam = 0.14
    magic = [2, 8, 20, 28, 50, 82, 126, 184]

    def find_nearest(lst, target):
        return min(lst, key=lambda x: abs(x - target))
    nup = np.array([abs(x - find_nearest(magic, x)) for x in Z])
    nun = np.array([abs(x - find_nearest(magic, x)) for x in N])
    P = nup*nun/(nup+nun)
    P[np.isnan(P)] = 0
    return alpham*P + betam*P**2

# Define a function to check cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 and norm_v2:
        similarity = dot_product / (norm_v1 * norm_v2)
    else:
        similarity = 0
    return similarity

# Define a function to check rms similarity
def rms_similarity(v1, v2):
    normv1 = np.sqrt(np.mean(((v1))**2))
    rms = np.sqrt(np.mean(((v1-v2))**2))
    return rms/normv1

def create_random_function(x, y, num_runs=100):
    for _ in range(num_runs):
        # Generate random operations
        ops = generate_random_ops()

        # Keep applying operations
        z = np.array(recursive_apply(x, y, ops))

        # Check if z is a NumPy array with numerical values and no nan or inf
        if isinstance(z, np.ndarray) and not np.any(np.isnan(z)) and not np.any(np.isinf(z)):
            return z

def generate_random_ops():
    # List of ops that maintain numeric stability
    ops = [np.add, np.subtract, np.multiply, np.divide, np.power, np.remainder,
           np.maximum, np.minimum, np.fmax, np.fmin,
           np.sin, np.cos, np.tan, np.arcsin, np.arctan,
           np.sqrt, np.square, np.exp, np.log10, np.log2,
           ]
    num_ops = random.randint(5, 15)
    return random.choices(ops, k=num_ops)

def recursive_apply(x, y, ops, depth=0, max_depth=15):
    if depth < len(ops) and depth < max_depth:
        op = ops[depth]
        result = recursive_apply(x, y, ops, depth + 1)

        if result is not None:
            try:
                z = op(result, y)
                if z is not None:
                    return z
            except TypeError:
                pass  # Handle the TypeError if needed

    # Return a default value (you can modify this as needed)
    return np.zeros_like(x)

def random_avg_similarity(v1, test):
    overlaps = []
    for _ in range(10):
        for _ in range(200):
            # Generate random function output
            v2 = create_random_function(Z, N)

            # Calculate cosine similarity
            if test == 'rms':
                sim = rms_similarity(v1,v2)
            elif test == 'cos':
                sim = cosine_similarity(v1,v2)
            overlaps.append(sim)

        # Check average similarity
        avg_sim = np.mean(overlaps)
        return avg_sim
    
# Defining the data-set
path = "spiral"
trainer = Trainer.from_path(path, which_folds=[0])

data = trainer.data
task_name = "binding_semf"
X, targets, _ = preds_targets_zn(0, data, task_name, train=True, val=True)
N=np.array(X[:,1])
Z=np.array(X[:,0])   

semf_terms = [volume_term,surface_term,coulomb_term,asymmetry_term,curvature_term,coulomb_exchange,wigner_term,surface_asym,delta,shell]


# Executing fitting routines and plots 

v1 = np.array(volume_term(Z, N))

v1poly = polyfit3d(Z, N, v1, 5, 5, subset_size=3)[0]
v1fourier = apply_fourier_2d(Z, N, v1, 5, 5)[0]

plot_rms_fit(Z, N, volume_term, 10, 1000, 'poly')

# Import PCA contributions

PCA_sums = np.load('scripts/last_layer/acts_pca.npy')
ZN = np.load('scripts/last_layer/zn.npy')

# Calculate the relative contributions of various PCAs to each term as well as the goodness of the fit as relative RMS

for term in semf_terms:
    linreg_PCA(ZN, PCA_sums, term)
