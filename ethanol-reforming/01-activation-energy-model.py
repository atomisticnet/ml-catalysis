#!/usr/bin/env python

"""
Construct ML Model 1 for predicting transition-state energies from
thermochemical DFT data and chemical information.

The model uses a combination of Random Forest Regression and Gaussian
Process Regression.

"""

import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

__author__ = "Nongnuch Artrith"
__email__ = "nartrith@atomistic.net"
__date__ = "2019-11-10"
__version__ = "0.1"


def plot_predictions(predictions, data):
    font = FontProperties()
    font.set_size(24)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.tick_params(labelsize=24)
    surfaces = set(data['Surface'])
    for s in surfaces:
        idx = (data['Surface'] == s)
        ax.scatter(predictions[idx], data[idx]['E_TS'], label=s, s=120)
    ideal = np.linspace(0, np.max(data['E_TS']), 10)
    ax.plot(ideal, ideal, color='black')
    ax.set_xlabel('Predicted Transition State (eV)', fontproperties=font)
    ax.set_ylabel('Reference Transition State (eV)', fontproperties=font)
    font.set_size(16)
    ax.legend(loc='lower right', prop=font)
    plt.savefig('validation-TS-model-RFR+GPR.png', bbox_inches='tight')
    plt.savefig('validation-TS-model-RFR+GPR.pdf', bbox_inches='tight')


def RFR(features, targets):
    model = RandomForestRegressor(
        max_depth=9, n_estimators=15, random_state=False, verbose=False)
    N = len(features)
    F = features
    T = targets.values
    predictions = []
    errors = []
    for i in range(N):
        idx = np.ones(N, dtype=bool)
        idx[i] = False
        model.fit(F[idx], T[idx])
        prediction_i = model.predict([F[i]])
        predictions.append(prediction_i)
        errors.append(prediction_i - T[i])
    predictions = np.array(predictions)
    rmse = np.std(errors)
    mae = np.mean(np.abs(errors))
    model.fit(F, T)
    return model, predictions, rmse, mae


def GPR(features, targets):
    kernel = RBF(length_scale=2.0, length_scale_bounds=(1e-05, 100000.0))
    model = GaussianProcessRegressor(
        kernel=kernel, alpha=0.05, random_state=False)
    N = len(features)
    F = features
    T = targets.values
    predictions = []
    errors = []
    for i in range(N):
        idx = np.ones(N, dtype=bool)
        idx[i] = False
        model.fit(F[idx], T[idx])
        prediction_i = model.predict([F[i]])
        predictions.append(prediction_i)
        errors.append(prediction_i - T[i])
    predictions = np.array(predictions)
    rmse = np.std(errors)
    mae = np.mean(np.abs(errors))
    model.fit(F, T)
    return model, predictions, rmse, mae


def model(dft_data):
    data = pd.read_csv(dft_data)
    features_all = data[['d_NN(top)', 'd_NN(2nd)', 'EN(top)', 'EN(2nd)',
                         'Facet', 'Eads(CH3CH2OH)', 'N_H', 'Initial', 'Final']]
    targets_all = data['E_TS']
    feature_scaler = StandardScaler().fit(
        np.array(features_all.values, dtype=np.float64))
    features_all = feature_scaler.transform(
        np.array(features_all.values, dtype=np.float64))
    select = (data['Facet'] == 1) & (data['has_TS'] == 1)
    features = features_all[select]
    targets = targets_all[select]

    # fit models and plot predictions
    model_rfr, predictions_rfr, rmse_rfr, mae_rfr = RFR(features, targets)
    model_gpr, predictions_gpr, rmse_gpr, mae_gpr = GPR(features, targets)
    predictions_mean = (predictions_rfr + predictions_gpr)/2.0
    plot_predictions(predictions_mean, data[select])

    # print out uncertainty estimate
    rmse = np.std(predictions_mean[:, 0] - targets.values)
    mae = np.std(np.abs(predictions_mean[:, 0] - targets.values))
    print("CV RMSE (RFR+GPR) = {}".format(rmse))
    print("CV MAE  (RFR+GPR) = {}".format(mae))

    # use model to predict unknown transition-state energies
    sel_predict = (data['Facet'] == 1) & (data['has_TS'] == 0)
    TS_predict_rfr = model_rfr.predict(features_all[sel_predict])
    TS_predict_gpr = model_gpr.predict(features_all[sel_predict])
    TS_predict = (TS_predict_rfr + TS_predict_gpr)/2
    data2 = data.copy()
    data2.loc[sel_predict, 'E_TS'] = TS_predict
    data2[sel_predict].to_csv('predicted-TS-RF+GPR.csv', index=False)


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "dft_data",
        help="CSV file with DFT data.",
        default="database-dft.csv",
        nargs="?")

    args = parser.parse_args()

    model(args.dft_data)
