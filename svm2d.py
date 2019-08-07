"""
Multiclass SVMs (Crammer-Singer formulation).
A pure Python re-implementation of:
Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex.
Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
ICPR 2014.
http://www.mblondel.org/publications/mblondel-icpr2014.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import stats
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pylab as pl

def projection_simplex(v, z=1):
    """
    Projection onto the simplex:
        w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
    # For other algorithms computing the same projection, see
    # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, gamma=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.gamma = gamma,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection onto the simplex.
        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize labels.
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Initialize primal and dual coefficients.
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))

        # Pre-compute norms.
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indices.
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]

                # All-zero samples can be safely ignored.
                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for the ith sample.
                delta = self._solve_subproblem(g, y, norms, i)

                # Update primal and dual coefficients.
                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print("iter", it + 1, "violation", vratio)

            if vratio < self.gamma:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    # START LOAD DATA TRAINING
    bankdata = pd.read_csv("D:/KULIAH/Digital Signal Processing/Identification Logat/data_training.csv")
    X = bankdata.drop(['filename', 'label'], axis=1)
    yy = bankdata['label']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(yy).astype('float64')
    # END LOAD DATA
    # START LOAD DATA TESTING
 #   bankdata2 = pd.read_csv("D:/KULIAH/Digital Signal Processing/Identification Logat/data_testing.csv")
 #   X_test = bankdata2.drop(['filename', 'label'], axis=1)
 #   label_test = bankdata2['label']

    X_train, X_test, label_train, label_test = train_test_split(X, y, test_size=0.50)

    # END LOAD DATA

    # START SELEKSI FITUR dengan PCA
    # Standart Deviasi
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    #END S.Deviasi

    #Start PCA
    # pca = PCA(n_components=25, whiten=True)
    # X_train_features = pca.fit_transform(X_train_std)
    # X_test_features = pca.transform(X_test_std)

    #End PCA


    pca = PCA(n_components=2).fit(X_train_std)
    pca_2d = pca.transform(X_train_std)

    svmClassifier_2d =  MulticlassSVM(C=0.1, gamma=0.01, max_iter=1000, random_state=0, verbose=1)
    svmClassifier_2d.fit(pca_2d, label_train)

    np.savetxt("D:/KULIAH/Digital Signal Processing/Identification Logat/pca2d.csv", pca_2d, delimiter=",")



    for i in range(0, pca_2d.shape[0]):
        if label_train[i] == 0:
            c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif label_train[i] == 1:
            c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif label_train[i] == 2:
            c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
        elif label_train[i] == 3:
            c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='^')

    pl.legend([c1, c2, c3, c4], ['banjar_hulu','banjar_kuala', 'dayak_bakumpai', 'dayak_ngaju'])
    x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
    y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
    Z =  svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
    zz = Z.reshape(xx.shape)
    pl.contour(xx, yy, zz)
    pl.title('Support Vector Machine Decision Surface')
    pl.axis('on')
    pl.show()

    print("Training set score for SVM: %f", svmClassifier_2d.score(pca_2d, label_train))