# =============================================================================
# Data processing
# Sourya Dey, USC
# =============================================================================

import numpy as np
from scipy import linalg


# =============================================================================
# =============================================================================
# # Normalization and feature extraction
# =============================================================================
# =============================================================================
def no_normalize(xtr,xva,xte):
    '''
    Dummy function to be used
    '''
    return xtr, xva, xte


def gaussian_normalize(xtr,xva,xte):
    '''
    Normalize features of training data by subtracting mean and dividing by sigma for each feature
    Test and val data are normalized using training mean and sigma
    xtr, xva, xte: Must be ndarrays of shape (numsamples,numfeatures)
    REASON: Normalizing results in cost function level sets being circular, so convergence is faster and local minima are avoided
    '''
    mu = np.average(xtr, axis=0).astype('float')
    xtr = xtr-mu
    sigma = np.std(xtr, axis=0).astype('float') #remember that subtracting mu first doesn't change sigma
    xtr = xtr/sigma
    if xva is not None and np.array_equal(xva,np.array([]))==0:
        xva = (xva-mu)/sigma
    if xte is not None and np.array_equal(xte,np.array([]))==0:
        xte = (xte-mu)/sigma
    return xtr, xva, xte


def minmax_normalize(xtr,xva,xte):
    '''
    Normalize features of training data by subtracting min and dividing by (max-min) for each feature
    Test and val data are normalized using training params
    '''
    mini = np.min(xtr, axis=0).astype('float')
    maxi = np.max(xtr, axis=0).astype('float')
    xtr = (xtr-mini)/(maxi-mini)
    if xva is not None and np.array_equal(xva,np.array([]))==0:
        xva = (xva-mini)/(maxi-mini)
    if xte is not None and np.array_equal(xte,np.array([]))==0:
        xte = (xte-mini)/(maxi-mini)
    return xtr,xva,xte


def zca_whiten(xtr,xva,xte, eps=1e-5):
    '''
    Compute ZCA whitening transformation as given in Krizhevsky - 'Learning Multiple layers of features from Tiny Images' Appendix A
    '''
    shape = xtr.shape
    xtr = xtr.reshape(xtr.shape[0],-1) #In case xtr shape was (n,d1,d2,...), now it's (n,d)
    mu = np.average(xtr, axis=0).astype('float') #mean of features (d,)
    xtr = xtr-mu

    cov = np.dot(xtr.T,xtr)/(xtr.shape[0]-1) #covariance matrix of features (d,d)
    u,s,_ = linalg.svd(cov) #SVD will be the same as eigendecomposition for the real,symmetric matrix cov
    Spmh = np.diag(1./np.sqrt(s+eps)) #s to the power minus half in matrix form. eps prevents numerical errors
    zca = np.dot(np.dot(u,Spmh),u.T) #The final ZCA whitening transformation matrix (d,d)

    xtr = np.dot(xtr,zca) #(n,d)
    xtr = xtr.reshape(shape) #(n,d1,d2,...)

    if xva is not None and np.array_equal(xva,np.array([]))==0:
        shape = xva.shape
        xva = xva.reshape(xva.shape[0],-1)
        xva = xva-mu
        xva = np.dot(xva,zca)
        xva = xva.reshape(shape)
    if xte is not None and np.array_equal(xte,np.array([]))==0:
        shape = xte.shape
        xte = xte.reshape(xte.shape[0],-1)
        xte = xte-mu
        xte = np.dot(xte,zca)
        xte = xte.reshape(shape)
    return xtr,xva,xte


def global_contrast_normalize(xtr,xva,xte, s=1,lamda=0,eps=1e-5):
    '''
    Every sample (from tr,va,te) is individually normalized using its mean and sigma
    This is different from gaussiana, minmax, zca_whiten:
        There, features are normalized by calculating (mean,sigma) across all inputs for each feature
        Here, inputs are normalized by calculating (mean,sigma) across all features for each input
    See DL book pgs 442-445 for s,lamda,eps and relevant concepts
    '''
    mu = np.average(xtr.reshape(xtr.shape[0],-1), axis=-1) #find mean of each image in xtr by flattening all of its pixels into 1 dimension
    mucast = mu[:,np.newaxis,np.newaxis,np.newaxis] #cast to have same shape as xtr
    var = np.var(xtr.reshape(xtr.shape[0],-1), axis=-1) #find variance of each image in xtr by flattening all of its pixels into 1 dimension
    varcast = var[:,np.newaxis,np.newaxis,np.newaxis] #cast to have same shape as xtr
    denom = np.maximum(eps, np.sqrt(lamda+varcast)) #denominator of new xtr
    xtr = s * (xtr-mucast)/denom #compute new xtr

    #Now do the same for xva and xte (steps are combined)
    if xva is not None and np.array_equal(xva,np.array([]))==0:
        xva = s * (xva - np.average(xva.reshape(xva.shape[0],-1), axis=-1)[:,np.newaxis,np.newaxis,np.newaxis]) / np.maximum(eps, np.sqrt(lamda+np.var(xva.reshape(xva.shape[0],-1), axis=-1)[:,np.newaxis,np.newaxis,np.newaxis]))
    if xte is not None and np.array_equal(xte,np.array([]))==0:
        xte = s * (xte - np.average(xte.reshape(xte.shape[0],-1), axis=-1)[:,np.newaxis,np.newaxis,np.newaxis]) / np.maximum(eps, np.sqrt(lamda+np.var(xte.reshape(xte.shape[0],-1), axis=-1)[:,np.newaxis,np.newaxis,np.newaxis]))
    return xtr,xva,xte


def principal_component_analysis(xtr,xva,xte, f2=20, centering=True, feature_shuffle=True):
    '''
    Given xtr with f1 features, compute PCA matrix pca of shape (f1,f2) by using SVD
    Post-multiply xtr,xva,xte with this pca (obtained from xtr alone) and return
    centering: If True, xtr features are first made mean 0
    feature_shuffle:
        If True, the final f2 features after applying PCA are shuffled
        This prevents the 1st neuron from being the most important, then the 2nd, and so on
    '''
    f1 = xtr.shape[1]
    if centering:
        xtr -= np.mean(xtr, axis=0)
    e,v = np.linalg.eig(xtr.T.dot(xtr)) #eigendecomposition of X^T*X gives SVD of X
    eigs = [(np.abs(e[i]),v[:,i]) for i in range(len(e))] #list of tuples (eigenvalue,eigenvector)
    eigs = sorted(eigs, key = lambda k : k[0], reverse=True) #sort according to eigenvalue, largest to smallest
    pca = np.real(np.hstack([eigs[i][1].reshape(f1,1) for i in range(f2)])) #get f2 eigenvectors corresponding to max f2 abs eigenvalues. The pca matrix is (f1,f2)

    xtr = xtr.dot(pca)
    xva = xva.dot(pca)
    xte = xte.dot(pca)
    if feature_shuffle:
        n = np.random.permutation(f2)
        xtr = xtr[:,n[:]]
        xva = xva[:,n[:]]
        xte = xte[:,n[:]]
    return xtr,xva,xte


def linear_discriminant_analysis(xtr,ytr, xva,xte, f2=20):
    '''
    Use linear discriminant analysis for dimensionality reduction from f1 original features to f2
    Follows approach here: http://sebastianraschka.com/Articles/2014_python_lda.html
        Computes a matrix lda of size f1xf2 using tr data
        Post-multiplies xtr,xva,xte by this
    IO:
        All xdata must be in (numsamples,numfeatures) form
        ydata must be one-hot encoded, i.e. (numsamples,numclasses)
        f2: No. of features to keep
    '''
    f1 = xtr.shape[1]
    c = ytr.shape[1]

    mu_overall = np.mean(xtr, axis=0)
    sc_intra = np.zeros((f1,f1))
    sc_inter = np.zeros((f1,f1))

    ## Compute scatter matrices ##
    for i in range(c):
        xtr_class = xtr[ytr[:,i]==1]
        count_class = len(xtr_class)
        mu_class = np.average(xtr_class, axis=0) #(numfeatures,)
        xtr_class_centered = xtr_class - mu_class #(numsamples_class,numfeatures)
        sc_intra += np.dot(xtr_class_centered.T,xtr_class_centered) #(numfeatures,numfeatures)
        mu_class_centered = mu_class - mu_overall #(numfeatures,)
        sc_inter += count_class*np.outer(mu_class_centered,mu_class_centered)

    ## Get new axes as eigenvectors with max abs eigenvalues ##
    if np.linalg.det(sc_intra) == 0:
        sc_final = np.linalg.pinv(sc_intra).dot(sc_inter) #Strictly speaking, pseudoinverse is not part of the actual LDA algorithm, but it's better than getting an error
    else:
        sc_final = np.linalg.inv(sc_intra).dot(sc_inter)
    e,v = np.linalg.eig(sc_final)
    eigs = [(np.abs(e[i]),v[:,i]) for i in range(len(e))] #list of tuples (eigenvalue,eigenvector)
    eigs = sorted(eigs, key = lambda k : k[0], reverse=True) #sort according to eigenvalue, largest to smallest
    lda = np.real(np.hstack([eigs[i][1].reshape(f1,1) for i in range(f2)])) #get f2 eigenvectors corresponding to max f2 abs eigenvalues. The lda matrix is (f1,f2)

    ## Transform feature space ##
    xtr = xtr.dot(lda)
    xva = xva.dot(lda)
    xte = xte.dot(lda)
    return xtr,xva,xte
# =============================================================================
# =============================================================================
