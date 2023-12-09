import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import lsqfit
from nuclr.data import get_nuclear_data, semi_empirical_mass_formula, BW_mass_formula
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def preds_targets_zn(model, data, task_name, train=True, val=True):
    # the data has an admittedly weird structure
    # data.X is a tensor of shape (N, 3) where N is the number of nuclei
    # TIMES the number of tasks. The first column is the number of protons,
    # the second column is the number of neutrons, and the third column is
    # the task index. 
    task_names = list(data.output_map.keys()) 
    
    if train and val:
        mask = torch.tensor([True for i in range(len(data[0]))])
    elif train:
        mask = data.train_masks[0]
    elif val:
        mask = data.val_masks[0] 
    task_idx = task_names.index(task_name)
    X_train = data.X[mask]
    
    tasks = X_train[:, 2].cpu().numpy()
    scatter = tasks == task_idx # get only rows relevant to task
    X_train_task = X_train[scatter][:,0:2]

    # get the targets and predictions for the task
    # first, we need to undo the preprocessing
    # data.regresion_transformer is a sklearn transformer that does the preprocessing
    # we can use its inverse_transform method to undo the preprocessing
    # it expects a numpy array, of shape (samples, features) where features is the number
    # of tasks we have.
    targets = data.y.view(-1, len(data.output_map.keys())).cpu().numpy()
    targets = data.regression_transformer.inverse_transform(targets)
    targets = targets.flatten()[mask.cpu().numpy()]
    targets = targets[scatter]

    # Predictions on the other hand are shape (samples, tasks)
    # each row has one correct prediction, and the rest are not useful
    # this is not optimal but not worth optimizing for now
    if model:
        preds = model(data.X[mask])
        preds = preds.cpu().detach().numpy()
        preds = data.regression_transformer.inverse_transform(preds)[scatter, task_idx]
    else:
        preds = np.zeros(len(targets))
    
    semf = semi_empirical_mass_formula(X_train_task[:, 0], X_train_task[:, 1]).cpu().numpy()

    if task_name == 'binding_semf':
        preds = preds + semf
        targets = targets + semf
    
    return X_train_task, targets, preds
   
    
def get_range_dat(X_task, targets, preds, Z_range, N_range, clear_nan=False):
    
    if clear_nan:
        ind = ~np.isnan(targets)
        X_task = X_task[ind]
        targets = targets[ind]
        preds = preds[ind]

    inputs_indices = [i for i,nuclei in enumerate(X_task) if nuclei[0].item() in Z_range and nuclei[1].item() in N_range]
    X_task = X_task[inputs_indices]
    targets = targets[inputs_indices]
    preds = preds[inputs_indices]
    
    return X_task, targets, preds
    
def find_local_minima_maxima(data):
    local_minima_maxima = []

    for i in range(1, len(data) - 2):
        if data[i - 1] < data[i] > data[i + 1]:
            local_minima_maxima.append([i, data[i], "max"])
        elif data[i - 1] > data[i] < data[i + 1]:
            local_minima_maxima.append([i, data[i], "min"])

    return local_minima_maxima

def calculate_PCA(embedding, modified_PCA, n):
    
    # Calculate the PCA components
    pca = PCA(n_components=n)
    pca.fit(embedding)
    PCA_embedding = pca.fit_transform(embedding)

    # Reconstruct the modified embedding using the modified PCA components
    if modified_PCA:
        embedding = pca.inverse_transform(modified_PCA)
    
    print("PCA:", pca.explained_variance_ratio_, "\n")
    return PCA_embedding, embedding

# def get_fit_embeddings(X, ):
    
#     if 
def get_nucl_range(nucl, embs, nucl_min, nucl_max, parity):
    # Extract the first column (nucl values) from the tensor

    # Create a mask based on the specified conditions
    if parity == 'all':
        mask = ((nucl >= nucl_min) & (nucl <= nucl_max)).squeeze()
        
    else:    
        mask = ((nucl >= nucl_min) & (nucl <= nucl_max) & (nucl % 2 == parity)).squeeze()

    # Apply the mask to filter rows
    filtered_embs = embs[mask]
    filtered_nucl = nucl[mask]

    return filtered_nucl, filtered_embs


def envelope(X, p):
  if type(X)==dict:
      [x] = X.values()
  else:
      x = X
  [A, x0, B, f, y0] = p.values()
  fun = A*(X-x0)**2+B*np.sin(f*X)-y0
  return fun

def polynomial(X, p):
  if type(X)==dict:
      [x] = X.values()
  else:
      x = X
  [a] = p.values()
  fun = 0
  for i in range(len(a)):
      fun += a[i]*(x)**i
  return fun  

def polyfit3d(x, y, z, kx, ky, order=None, subset_size=None):

    # Use a subset of the data if specified
    if subset_size is not None and subset_size < len(x):
        indices = np.random.choice(len(x), size=subset_size, replace=False)
        x_sub = x[indices]
        y_sub = y[indices]
        z_sub = z[indices]
    else:
        x_sub = x
        y_sub = y
        z_sub = z

    # grid coords
    x_grid, y_grid = np.meshgrid(x_sub, y_sub)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx + 1, ky + 1))

    # solve array
    a = np.zeros((coeffs.size, x_sub.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x_sub)
        else:
            arr = coeffs[i, j] * x_sub**i * y_sub**j
        a[index] = arr

    # do leastsq fitting and return leastsq result
    result = np.linalg.lstsq(a.T, z_sub, rcond=None)

    # Get the coefficients from the result
    coefficients = result[0]

    # Define the fitted surface function
    terms = [coeff * x**i * y**j for (j, i), coeff in np.ndenumerate(coefficients.reshape((kx + 1, ky + 1)))]
    return np.sum(terms, axis=0), coefficients

def fourier_series_2d(x, y, coeffs, Nx, Ny, subset_size=None):
    """
    Fourier series function for two variables with independent frequencies.
    x, y: input variables
    coeffs: coefficients of the Fourier series
    Nx, Ny: number of harmonics in x and y dimensions
    """
    a0 = coeffs[0]
    result = a0
    coeff_idx = 1  # Index for accessing elements in coeffs

    # Use a subset of the data if specified
    if subset_size is not None and subset_size < len(x):
        indices = np.random.choice(len(x), size=subset_size, replace=False)
        x_sub = x[indices]
        y_sub = y[indices]

    else:
        x_sub = x
        y_sub = y

    for nx in range(1, Nx + 1):
        for ny in range(1, Ny + 1):
            # Extract coefficients for current harmonics
            freq_x = coeffs[coeff_idx]
            freq_y = coeffs[coeff_idx + 1]
            an = coeffs[coeff_idx + 2]
            bn = coeffs[coeff_idx + 3]

            result += an * np.cos(nx * freq_x * x_sub) * np.cos(ny * freq_y * y_sub) \
                    + bn * np.sin(nx * freq_x * x_sub) * np.sin(ny * freq_y * y_sub)

            coeff_idx += 4  # Move to the next set of coefficients

    return result

def apply_fourier_2d(X, Y, Z, Nx, Ny, subset_size=None):
    """
    Apply Fourier series fitting for two-variable function with frequencies.
    X, Y: input variables
    Z: target output
    Nx, Ny: number of harmonics in x and y dimensions
    """
    # Number of coefficients: 1 (a0) + 4 * Nx * Ny (including frequencies)
    num_coeffs = 1 + 4 * Nx * Ny
    initial_guess = [0.01] * num_coeffs  # Initial guess for frequencies and coefficients

    # Flatten X, Y, Z for curve_fit
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Fit Fourier series
    params, params_covariance = curve_fit(
        lambda x_y, *coeffs: fourier_series_2d(x_y[0], x_y[1], coeffs, Nx, Ny, subset_size),
        (X, Y),
        Z,
        p0=initial_guess
    )

    # Use the optimized coefficients to reconstruct Z
    Z_fit = fourier_series_2d(X, Y, params, Nx, Ny)

    return Z_fit, params, params_covariance

    
def PCA_fit(X, y, fit_func, n_pol):
    
  magic_numbers = [2, 8, 20, 28, 50, 82, 126]
  # prior = {"a": gvar.gvar([0.065], [0.1])}      
  
  if fit_func == polynomial:
      p0 = {"a": [0.4]*(n_pol+1)}
  elif fit_func == envelope:                 
      p0 = {"A": [2*10**(-4)], "x0": [70], "B": [0.35], "f":[0.2], "y0":[0.8]}
  
  shape = y.shape[0]
  cov = np.zeros((shape,shape), dtype=int)
  for i in range(shape):
     if X[i] in magic_numbers:
         cov[i, i] = 4
     else:
         cov[i, i] = 1
  return lsqfit.nonlinear_fit(data=(X, y, cov), fcn=fit_func, p0=p0, svdcut=1e-12)

def mask_uncertainities(min_included_nucl, task_name):
    df = get_nuclear_data(False)
    
    df = df[
        (df.z > min_included_nucl) & (df.n > min_included_nucl)
    ]
    
    if task_name=="binding_semf":
        mask = np.logical_or((df.binding_unc * (df.z + df.n) < 100).values,~(df.binding_sys == 'Y').values)
    elif task_name=="radius":
        mask = (df.unc_r < 0.005).values
    return mask


def rms_val(X, targets, preds, task_name, mask_unc=None):  
    
    # returns the value of rms in keV
    
    A = np.array(X[:,0]+X[:,1])
    mask = ~np.logical_or(np.isnan(targets),np.isnan(preds))
    
    if mask_unc is not None:
        mask = np.logical_and(mask,mask_unc)
    
    targets = targets[mask]
    preds = preds[mask]    
    A = A[mask]

    if task_name=="binding_semf" or task_name=="binding" :
        return np.sqrt(np.mean((A*(targets-preds))**2))
    else:
        return np.sqrt(np.mean(((targets-preds))**2))


def plot_rms_fit(Z, N, term_function, order, subset_size, fit):
    rms_similarities = []

    for i in range(order):
        v1 = term_function(Z, N)

        if fit =='poly':
            v2 = polyfit3d(Z, N, v1, i, i, i, subset_size)
        elif fit == 'fourier':
            v2 = apply_fourier_2d(Z, N, v1, i, i)[0]

        rms_similarity_value = np.sqrt(np.mean(((v1-v2))**2))
        rms_similarities.append(rms_similarity_value)

    plt.scatter(np.arange(order), rms_similarities, label=f'RMS for {term_function.__name__}')
    plt.xlabel('order')
    plt.ylabel('RMS')
    plt.legend()
    plt.show()


def linreg_PCA(ZN, PCA_sums, term_function):
    # Create new vectors as specified
    PCA = [PCA_sums[0]]  
    for i in range(1, len(PCA_sums)):
        PCA.append(PCA_sums[i] - np.sum(PCA[:i], axis=0))

    Z = ZN[:,0]
    N = ZN[:,1]
    
    # Prepare the input matrix for linear regression
    X = np.array(PCA)  
    y = term_function(Z, N)  # the target for linear regression

    # Perform linear regression
    reg = LinearRegression().fit(X, y)

    y_pred = reg.predict(X)

    relrms = 100*np.sqrt(np.mean(((y_pred-y))**2))/np.sqrt(np.mean(((y))**2))


    # Print the coefficients
    coefficients = reg.coef_
    contributions = np.array([coefficients[i]*X[:,i] for i in range(len(coefficients))])

    norm_contributions = np.array([np.linalg.norm(contributions[i]) for i in range(len(coefficients))])

    relative_contributions = 100*norm_contributions / np.sum(norm_contributions)
    intercept = reg.intercept_

    # Display the coefficients and the relative RMS
    print(f'Relative contribution % of PCAs for {term_function.__name__}:', relative_contributions)
    print("Relative % RMS:", relrms)
