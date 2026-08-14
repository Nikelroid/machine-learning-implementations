import numpy as np
from sklearn.decomposition import PCA

def pca_approx(M, m=100):
    np.random.seed(12) 
    M_tilde = np.log(1 + M)
    Mc = M_tilde - np.mean(M_tilde, axis=0) 
    model = PCA(n_components=m) 
    model.fit(Mc) 
    V = model.components_.T
    eigenvalues = model.explained_variance_ 
    frac_var = np.sum(eigenvalues[:m]) / np.sum(eigenvalues) 
    return Mc, V, eigenvalues, frac_var

def compute_embedding(Mc, V):
    P = Mc @ V 
    E = P / np.linalg.norm(P, axis=0) 
    E = E / np.linalg.norm(E, axis=1, keepdims=True) 
    return E