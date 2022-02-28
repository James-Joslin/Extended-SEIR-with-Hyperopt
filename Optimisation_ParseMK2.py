import os
import sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.getOrCreate()

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, SparkTrials, Trials
from numpy import linalg, poly1d, polyfit
from seirsplus.models import ExtSEIRSNetworkModel
from seirsplus.networks import generate_demographic_contact_network
from seirsplus.utilities import gamma_dist

import json

N = 30000
SurreyPop = 1160000

cases_df = pd.read_csv("./CasesGovUK.csv")
cases_df = cases_df.iloc[1:189]
cases_df = cases_df.sort_values("date")
print(cases_df.head(9))

INIT_INFECTED = (sum(cases_df['Cases'].iloc[182:189]) / SurreyPop) * N
INIT_HOSP = 80/SurreyPop * N
deaths_df = pd.read_csv("./DeathsGovUK.csv")
deaths_df = deaths_df.iloc[1:182]
deaths_df['date'] = pd.to_datetime(deaths_df['date'], dayfirst=True)
deaths_df = deaths_df.sort_values("date")
deaths_df['CumDeaths'] = deaths_df['NewDeaths'].cumsum()
deaths_df = deaths_df.reset_index(drop=True)
deaths_df['T'] = deaths_df.index
# print(deaths_df)

UK_data = {
    'age_distn': {
        '0-9': 0.11800000000000001, '10-19': 0.095, '20-29': 0.162,
        '30-39': 0.133, '40-49': 0.146, '50-59': 0.121, '60-69': 0.10800000000000001,
        '70-79': 0.071, '80+': 0.046},
    'household_size_distn': {1: 0.273, 2: 0.344, 3: 0.159, 4: 0.157, 5: 0.050, 6: 0.01, 7: 0.007},
    'household_stats': {'pct_with_under20': 0.447,  # percent of households with at least one member under 60
                        'pct_with_over60': 0.380,  # percent of households with at least one member over 60
                        'pct_with_under20_over60': 0.034,
                        # percent of households with at least one member under 20 and at least one member over 60
                        'pct_with_over60_givenSingleOccupant': 0.273,
                        # percent of households with a single-occupant that is over 60
                        'mean_num_under20_givenAtLeastOneUnder20': 1.91
                        # number of people under 20 in households with at least one member under 20
                        }
}

demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
    N=N, demographic_data=UK_data,
    distancing_scales=[],
    isolation_groups=[])

G_baseline = demographic_graphs['baseline']

households_indices = [household['indices'] for household in households]

ageGroup_pctHospitalized = {'0-9': float(17/378),
                            '10-19': float(11/378),
                            '20-29': float(35/378),
                            '30-39': float(36/378),
                            '40-49': float(46/378),
                            '50-59': float(43/378),
                            '60-69': float(53/378),
                            '70-79': float(61/378),
                            '80+': float(76/378)}
PCT_HOSPITALIZED = [ageGroup_pctHospitalized[ageGroup] for ageGroup in individual_ageGroups]

ageGroup_hospitalFatalityRate = {'0-9': 0.0000,
                                 '10-19': 0.4527,
                                 '20-29': 0.0577,
                                 '30-39': 0.0426,
                                 '40-49': 0.0694,
                                 '50-59': 0.1532,
                                 '60-69': 0.3381,
                                 '70-79': 0.5187,
                                 '80+': 0.7283}
PCT_FATALITY = [ageGroup_hospitalFatalityRate[ageGroup] for ageGroup in individual_ageGroups]

def run_hypOpt(params):
    params.update({'A1': max(params['A1'], 0)})
    params.update({'A2': max(params['A2'], 0)})
    params.update({'A3': max(params['A3'], 0)})
    params.update({'A4': max(params['A4'], 0)})
    params.update({'A5': max(params['A5'], 0)})
    params.update({'A6': max(params['A6'], 0)})
    params.update({'A7': max(params['A7'], 0)})
    params.update({'A8': max(params['A8'], 0)})
    params.update({'A9': max(params['A9'], 0)})

    params.update({'lP_m': max(params['lP_m'], 0)})
    params.update({'lP_c': max(params['lP_c'], 0)})
    params.update({'psP_m': max(params['psP_m'], 0)})
    params.update({'psP_c': max(params['psP_c'], 0)})
    params.update({'sP_m': max(params['sP_m'], 0)})
    params.update({'sP_c': max(params['sP_c'], 0)})
    params.update({'oHP_m': max(params['oHP_m'], 0)})
    params.update({'oHP_c': max(params['oHP_c'], 0)})
    params.update({'hDP_m': max(params['hDP_m'], 0)})
    params.update({'hDP_c': max(params['hDP_c'], 0)})
    params.update({'hDeP_m': max(params['hDeP_m'], 0)})
    params.update({'hDeP_c': max(params['hDeP_c'], 0)})
    params.update({'p_asym': max(params['p_asym'], 0)})
    params.update({'r0_c': max(params['r0_c'], 0)})
    params.update({'betaMUL': max(params['betaMUL'], 0)})
    params.update({'P_GLOB': max(params['P_GLOB'], 0)})

    ALPHA = []
    for z in range(len(individual_ageGroups)):
        if individual_ageGroups[z] == '0-9':
            ALPHA.append(float(params['A1']))
        elif individual_ageGroups[z] == '10-19':
            ALPHA.append(float(params['A2']))
        elif individual_ageGroups[z] == '20-29':
            ALPHA.append(float(params['A3']))
        elif individual_ageGroups[z] == '30-39':
            ALPHA.append(float(params['A4']))
        elif individual_ageGroups[z] == '40-49':
            ALPHA.append(float(params['A5']))
        elif individual_ageGroups[z] == '50-59':
            ALPHA.append(float(params['A6']))
        elif individual_ageGroups[z] == '60-69':
            ALPHA.append(float(params['A7']))
        elif individual_ageGroups[z] == '70-79':
            ALPHA.append(float(params['A8']))
        else:
            ALPHA.append(float(params['A9']))

    latentPeriod_mean = float(params['lP_m'])
    latentPeriod_coeffvar = float(params['lP_c'])

    SIGMA = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)

    presymptomaticPeriod_mean = float(params['psP_m'])
    presymptomaticPeriod_coeffvar = float(params['psP_c'])

    LAMDA = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)
    # dist_info([1/LAMDA, 1/SIGMA, 1/LAMDA+1/SIGMA], ["latent period", "pre-symptomatic period", "total incubation period"], plot=True, colors=['gold', 'darkorange', 'black'], reverse_plot=True)

    symptomaticPeriod_mean = float(params['sP_m'])
    symptomaticPeriod_coeffvar = float(params['sP_c'])

    GAMMA = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)
    infectiousPeriod = 1 / LAMDA + 1 / GAMMA
    # dist_info([1/LAMDA, 1/GAMMA, 1/LAMDA+1/GAMMA], ["pre-symptomatic period", "(a)symptomatic period", "total infectious period"], plot=True, colors=['darkorange', 'crimson', 'black'], reverse_plot=True)

    onsetToHospitalizationPeriod_mean = float(params['oHP_m'])
    onsetToHospitalizationPeriod_coeffvar = float(params['oHP_c'])

    ETA = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

    hospitalizationToDischargePeriod_mean = float(params['hDP_m'])
    hospitalizationToDischargePeriod_coeffvar = float(params['hDP_c'])

    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)
    # dist_info([1/ETA, 1/GAMMA_H, 1/ETA+1/GAMMA_H], ["onset-to-hospitalization period", "hospitalization-to-discharge period", "onset-to-discharge period"], plot=True, colors=['crimson', 'violet', 'black'], reverse_plot=True)

    hospitalizationToDeathPeriod_mean = float(params['hDeP_m'])
    hospitalizationToDeathPeriod_coeffvar = float(params['hDeP_c'])

    MU_H = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)
    # dist_info([1/ETA, 1/MU_H, 1/ETA+1/MU_H], ["onset-to-hospitalization period", "hospitalization-to-death period", "onset-to-death period"], plot=True, colors=['crimson', 'darkgray', 'black'], reverse_plot=True)

    PCT_ASYMPTOMATIC = float(params['p_asym'])
    PCT_ASYMPTOMATIC = [0.80 if age in ['0-9', '10-19'] else PCT_ASYMPTOMATIC for age in individual_ageGroups]
    
    # Ugly and not at all elegant, but this means that you only need to change the past R values within the list above
    R0_coeffvar = float(params['r0_c'])
    R_Values = [1.45, 1.35, 0.9, 1.0, 0.8, 1.05, 1.2, 1.1, 0.95, 1.1, 1.1, 1.15, 0.9]
    R0 = gamma_dist(R_Values[0], R0_coeffvar, N)
    R1 = gamma_dist(R_Values[1], R0_coeffvar, N)
    R2 = gamma_dist(R_Values[2], R0_coeffvar, N)
    R3 = gamma_dist(R_Values[3], R0_coeffvar, N)
    R4 = gamma_dist(R_Values[4], R0_coeffvar, N)
    R5 = gamma_dist(R_Values[5], R0_coeffvar, N)
    R6 = gamma_dist(R_Values[6], R0_coeffvar, N)
    R7 = gamma_dist(R_Values[7], R0_coeffvar, N)
    R8 = gamma_dist(R_Values[8], R0_coeffvar, N)
    R9 = gamma_dist(R_Values[9], R0_coeffvar, N)
    R10 = gamma_dist(R_Values[10], R0_coeffvar, N)
    R11 = gamma_dist(R_Values[11], R0_coeffvar, N)
    R12 = gamma_dist(R_Values[12], R0_coeffvar, N)

    BETA0 = (1 / infectiousPeriod * R0) 
    BETA1 = (1 / infectiousPeriod * R1) 
    BETA2 = (1 / infectiousPeriod * R2) 
    BETA3 = (1 / infectiousPeriod * R3)  
    BETA4 = (1 / infectiousPeriod * R4)  
    BETA5 = (1 / infectiousPeriod * R5)  
    BETA6 = (1 / infectiousPeriod * R6)  
    BETA7 = (1 / infectiousPeriod * R7)  
    BETA8 = (1 / infectiousPeriod * R8) 
    BETA9 = (1 / infectiousPeriod * R9) 
    BETA10 = (1 / infectiousPeriod * R10) 
    BETA11 = (1 / infectiousPeriod * R11) 
    BETA12 = (1 / infectiousPeriod * R12) 

    BETA_PAIRWISE_MODE = 'infected'
    DELTA_PAIRWISE_MODE = 'mean'

    P_GLOBALINTXN = float(params['P_GLOB'])
    model_tseries = []
    model_farrays = []

    for i in range(0, 3, 1):
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN, 
                                     beta=BETA0, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE,
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE, initH=INIT_HOSP,
                                     G_Q=G_baseline, q=1, beta_Q=BETA0, isolation_time=0,
                                     initI_sym=(INIT_INFECTED * 0.3), initI_asym=(INIT_INFECTED * 0.7))
        checkpoints = {
            't' : [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168],
            'beta' : [BETA1, BETA2, BETA3, BETA4, BETA5, BETA6, BETA7, BETA8, BETA9, BETA10, BETA11, BETA12],
            'beta_Q' : [BETA1, BETA2, BETA3, BETA4, BETA5, BETA6, BETA7, BETA8, BETA9, BETA10, BETA11, BETA12]}
        model.run(T=182, checkpoints=checkpoints)
        model_tseries.append(pd.DataFrame(model.tseries))
        model_farrays.append(pd.DataFrame(model.numF))

    tseries_df = pd.concat(model_tseries)

    f_df = pd.concat(model_farrays)
    f_df[0] = f_df[0] * SurreyPop / N
    prediction_df = pd.DataFrame({'T': list(tseries_df[0]), 'F': list(f_df[0])})

    x = np.array(prediction_df['T'])
    y = np.array(prediction_df['F'])

    degree = 3
    z = polyfit(x, y, degree)
    f = poly1d(z)

    steps = int(np.amax(x))
    x_new = np.linspace(0, steps, steps)
    f_smoothed = f(x_new)

    obs_array = np.array(deaths_df['CumDeaths'])

    if len(f_smoothed) < len(obs_array):
        f_additional = np.repeat(np.amax(f_smoothed), int(len(obs_array) - len(f_smoothed)))
        f_smoothed = np.concatenate([f_smoothed, f_additional])
    if len(f_smoothed) > len(obs_array):
        f_smoothed = f_smoothed[0:int(len(obs_array))]
    for i in range(0, len(f_smoothed) - 1, 1):
        val_a = f_smoothed[i]
        val_b = f_smoothed[i + 1]
        if val_b < val_a:
            f_smoothed[i + 1] = val_a
    for i in range(len(f_smoothed)):
        if f_smoothed[i] < 0:
            f_smoothed[i] = 0

    plt.plot(obs_array)
    plt.plot(f_smoothed)
    plt.scatter(x = prediction_df['T'], y = prediction_df['F'])
    plt.show(block = False)
    plt.pause(4)
    plt.close()

    loss = linalg.norm(obs_array - f_smoothed)
    print(params, loss)
    return {'loss': loss, 'status': STATUS_OK}

search_space = {
    'A1': hp.quniform('A1', .55, .85, 0.01),
    'A2': hp.quniform('A2', .55, .85, 0.01),
    'A3': hp.quniform('A3', .55, .85, 0.01),
    'A4': hp.quniform('A4', .55, .85, 0.01),
    'A5': hp.quniform('A5', .55, .85, 0.01),
    'A6': hp.quniform('A6', .55, .85, 0.01),
    'A7': hp.quniform('A7', .55, .85, 0.01),
    'A8': hp.quniform('A8', .55, .85, 0.01),
    'A9': hp.quniform('A9', .55, .85, 0.01),

    'lP_m': hp.quniform('lP_m', 1.5, 3.5, 0.01),
    'lP_c': hp.quniform('lP_c', 0.55, 0.7, 0.01),

    'psP_m': hp.quniform('psP_m', 1.5, 3.5, 0.01),
    'psP_c': hp.quniform('psP_c', 0.25, 0.45, 0.01),

    'sP_m': hp.quniform('sP_m', 3.5, 12.5, 0.01),
    'sP_c': hp.quniform('sP_c', 0.25, .35, 0.01),

    'oHP_m': hp.quniform('oHP_m', 3, 6.5, 0.01),
    'oHP_c': hp.quniform('oHP_c', 0.45, 0.55, 0.01),

    'hDP_m': hp.quniform('hDP_m', 1.5, 5.5, 0.01),
    'hDP_c': hp.quniform('hDP_c', 0.2, 0.3, 0.01),

    'hDeP_m': hp.quniform('hDeP_m', 8.5, 12, 0.01),
    'hDeP_c': hp.quniform('hDeP_c', 0.2, 0.3, 0.01),

    'p_asym': hp.quniform('p_asym', 0.3, 0.8, 0.01),

    'r0_c': hp.quniform('r0_c', 0.5, 0.65, 0.01),

    'P_GLOB': hp.quniform('P_GLOB', 0.45, 0.55, 0.01)
}
num_epochs = int(int(os.cpu_count())*30)
with mlflow.start_run():
    best_hyperparameters = fmin(
        fn=run_hypOpt,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_epochs,
        trials=SparkTrials(parallelism=int(os.cpu_count()))
    )

results = dict(best_hyperparameters)
print(results)
with open('BestParameters.txt', 'w') as file:
     file.write(json.dumps(results))
file.close()