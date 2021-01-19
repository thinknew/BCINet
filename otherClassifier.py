'''
BCINet
© Avinash K Singh 
https://github.com/thinknew/bcinet
Licensed under MIT License
'''

import numpy as np
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error,f1_score, accuracy_score,average_precision_score,\
    precision_score,recall_score,mean_absolute_error,cohen_kappa_score
import time
from tensorflow.keras import utils as np_utils


def Classifiers(X,Y,f1_avg, numOfClasses):

    mse=[]
    ct=[]
    acc=[]
    f1_sc=[]
    mae = []
    avg_pre_sco = []
    co_kap_sco = []
    precision = []
    recall = []

    repo=np.empty(shape=(3,))
    estimators = [('LogRed', LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)),
                  ('LDA', LinearDiscriminantAnalysis()),
                  ('Linear SVM', SVC(kernel='linear')),
                  ('RBF SVM', SVC(kernel='rbf', gamma='scale')),
                  ('NN', MLPClassifier(alpha=1, max_iter=1000))]


    covest = Covariances(estimator='lwf')

    ts = TangentSpace()
    for name, estimator in estimators:
        model = make_pipeline(covest, ts, estimator)
        tic = time.clock()
        pred= cross_val_predict(model, X, Y, cv=2)
        toc = time.clock()
        mse.append(mean_squared_error(Y, pred))
        mae.append(mean_absolute_error(Y, pred))
        co_kap_sco.append(cohen_kappa_score(Y, pred))


        y = np_utils.to_categorical(Y.ravel(),numOfClasses)
        pred = np_utils.to_categorical(pred.ravel(),numOfClasses)

        if numOfClasses==2:
            y=y.argmax(axis=-1)
            pred=pred.argmax(axis=-1)
            acc.append(accuracy_score(y, pred))
            avg_pre_sco.append(average_precision_score(y, pred, average='weighted', pos_label=0))
            precision.append(precision_score(y, pred, average=f1_avg, pos_label=0))
            recall.append(recall_score(y, pred, average=f1_avg, pos_label=0))
            f1_sc.append(f1_score(y, pred, average=f1_avg, pos_label=0))
        else:
 
            acc.append(accuracy_score(y, pred))
            avg_pre_sco.append(average_precision_score(y, pred, average='weighted'))
            precision.append(precision_score(y, pred, average=f1_avg))
            recall.append(recall_score(y, pred, average=f1_avg))
            f1_sc.append(f1_score(y, pred, average=f1_avg))

        ct.append(toc - tic)



    return acc, mse, mae, avg_pre_sco, co_kap_sco, precision, recall, f1_sc, ct
