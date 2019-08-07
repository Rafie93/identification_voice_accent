"""
#Project Thesis By Rafie PCA-SVM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns; sns.set()


def projection_simplex(v, z=1):
    """
    Projection  simplex:
        w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
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

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]

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
                h=vratio
               #print("iter", it + 1, "violation", vratio)

            if vratio < self.tol:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements
    pass


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    # START LOAD DATA TRAINING
    bankdata = pd.read_csv("data_training.csv")
    X_train = bankdata.drop(['filename', 'label'], axis=1)
    label_train = bankdata['label']
    # END LOAD DATA
    # START LOAD DATA TESTING
    bankdata2 = pd.read_csv("data_testing.csv")
    X_test = bankdata2.drop(['filename', 'label'], axis=1)
    label_test = bankdata2['label']

#    X_train, X_test, label_train, label_test = train_test_split(X, y, test_size=0.50)

    # END LOAD DATA

    # START SELEKSI FITUR dengan PCA
    # Standart Deviasi
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    #END S.Deviasi

    #Start PCA
    pca = PCA(n_components=25, whiten=True)
    X_train_features = pca.fit_transform(X_train_std)
    X_test_features = pca.transform(X_test_std)

    #End PCA

    nC=0.6
    nTol = 0.01

    clf = MulticlassSVM(C=nC, tol=nTol, max_iter=1000, random_state=0, verbose=1)
    clf.fit(X_train_features, label_train)
    y_pred = clf.predict(X_test_features)

    pcsvm = MulticlassSVM(C=nC, tol=nTol, max_iter=1000, random_state=0, verbose=1)
    pcsvm.fit(X_train_std, label_train)
    predik= pcsvm.predict(X_test_std)

    # print(predik)

    joblib.dump(pcsvm, 'my_model.pkl')

 #   print("Training set score for PCA-SVM: %f", clf.score(X_train_features, label_train))
 #   print("Training set score for SVM: %f", svm.score(X_train_std, label_train))
    mcm2 = multilabel_confusion_matrix(label_test, y_pred)
    print(mcm2)
    akk = []
    for ii in mcm2:
        tP = (ii[0])
        fP = (ii[1])
        fN = (ii[2])
        tN = (ii[3])
        akurasi_klass = (tP + tN) / (tP + tN + fP + fN)
        akk.append(akurasi_klass)
    #   print("Akurasi Kelas ", i, " ", akurasi_klass * 100)

    print("Akurasi Multiclass Confusion Matrix SVM", (sum(akk) / 4)*100)

    mcm = multilabel_confusion_matrix(label_test, predik)
    print(mcm)
    ak = []
    for i in mcm:
        tP = (i[0])
        fP = (i[1])
        fN = (i[2])
        tN = (i[3])
        akurasi_klass = (tP + tN) / (tP + tN + fP + fN)
        ak.append(akurasi_klass)
     #   print("Akurasi Kelas ", i, " ", akurasi_klass * 100)

    print("Akurasi Multiclass Confusion Matrix PCA-SVM", (sum(ak) / 4)*100)