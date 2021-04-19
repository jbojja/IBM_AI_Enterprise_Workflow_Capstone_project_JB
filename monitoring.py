import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
import pandas as pd
from dataingestion2 import fetch_ts, engineer_features
localpath = os.path.abspath('')
data_dir = os.path.join(localpath,"data_dir")
latest_data_path = data_dir = os.path.join(data_dir,"ts-data")

def get_latest_train_data():
    """
    load the data used in the latest training
    """
    df = pd.read_csv(os.path.join(latest_data_path,'ts-all.csv'))
    return(df)

def get_monitoring_tools(df):
    """
    determine outlier and distance thresholds
    return thresholds, outlier model(s) and source distributions for distances
    NOTE: for classification the outlier detection on y is not needed

    """    
    X,y,dates = engineer_features(df)
    X1 = X.to_numpy()
    xpipe = Pipeline(steps=[('pca', PCA(2)),('clf', EllipticEnvelope(random_state=0,contamination=0.01))])
    xpipe.fit(X1)
    bs_samples = 549
    outliers_X = np.zeros(bs_samples)
    wasserstein_X = np.zeros(bs_samples)
    wasserstein_y = np.zeros(bs_samples)
    for b in range(bs_samples):
        n_samples = int(np.round(0.80 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X1.shape[0]),n_samples,replace=True).astype(int)
        y_bs=y[subset_indices]
        X_bs=X1[subset_indices,:]
    
        test1 = xpipe.predict(X_bs)
        wasserstein_X[b] = wasserstein_distance(X1.flatten(),X_bs.flatten())
        wasserstein_y[b] = wasserstein_distance(y,y_bs.flatten())
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))
        
    outliers_X.sort()
    outlier_X_threshold = outliers_X[int(0.975*bs_samples)] + outliers_X[int(0.025*bs_samples)]

    wasserstein_X.sort()
    wasserstein_X_threshold = wasserstein_X[int(0.975*bs_samples)] + wasserstein_X[int(0.025*bs_samples)]

    wasserstein_y.sort()
    wasserstein_y_threshold = wasserstein_y[int(0.975*bs_samples)] + wasserstein_y[int(0.025*bs_samples)]
    
    to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_X":np.round(wasserstein_X_threshold,2),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "clf_X":xpipe,
                 "X_source":X1,
                 "y_source":y,
                 "latest_X":X,
                 "latest_y":y}
    return(to_return)
