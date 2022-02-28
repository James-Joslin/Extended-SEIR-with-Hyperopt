import pandas as pd
from seirsplus.models import ExtSEIRSNetworkModel
from seirsplus.networks import generate_demographic_contact_network
from seirsplus.utilities import gamma_dist

def SEIR_Model(Num):
    latentPeriod_mean = 2.29
    latentPeriod_coeffvar = 0.63

    SIGMA = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)

    presymptomaticPeriod_mean = 3.41
    presymptomaticPeriod_coeffvar = 0.29

    LAMDA = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)

    symptomaticPeriod_mean = 3.5
    symptomaticPeriod_coeffvar = 0.29

    GAMMA = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)
    infectiousPeriod = 1 / LAMDA + 1 / GAMMA

    onsetToHospitalizationPeriod_mean = 5.51
    onsetToHospitalizationPeriod_coeffvar = 0.48

    ETA = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

    hospitalizationToDischargePeriod_mean = 5.32
    hospitalizationToDischargePeriod_coeffvar = 0.3

    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)

    hospitalizationToDeathPeriod_mean = 10.04
    hospitalizationToDeathPeriod_coeffvar = 0.22

    MU_H = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

    PCT_ASYMPTOMATIC = 0.8
    PCT_ASYMPTOMATIC = [0.80 if age in ['0-9', '10-19'] else PCT_ASYMPTOMATIC for age in individual_ageGroups]

    BETA_PAIRWISE_MODE = 'infected'
    DELTA_PAIRWISE_MODE = 'mean'
    P_GLOBALINTXN = 0.53

    R0_coeffvar = 0.6
    R_Values = [1.1, 1.35, 1.26, 1.35, 1.39, 1.3, 1.25, 1.31, 1.35, 1.33, 1.4, 1.4, 1.45]
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

    model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                 beta=BETA0, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                 gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                 a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                 alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE,
                                 delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                 G_Q=G_baseline, q=0, beta_Q=BETA0, isolation_time=0,
                                 initI_pre=INIT_PreSym, initH=((470/SurreyPop)*N),
                                 initI_asym=INIT_Iasym, initI_sym=INIT_Isym)

    checkpoints = {
            't' : [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168],
            'beta' : [BETA1, BETA2, BETA3, BETA4, BETA5, BETA6, BETA7, BETA8, BETA9, BETA10, BETA11, BETA12],
            'beta_Q' : [BETA1, BETA2, BETA3, BETA4, BETA5, BETA6, BETA7, BETA8, BETA9, BETA10, BETA11, BETA12]}
    model.run(T=180, checkpoints=checkpoints)
    df_sir = pd.DataFrame({
        'T': model.tseries, 'F': model.numF,
        'H': model.numH, 'Iasym': model.numI_asym, 
        'Isym': model.numI_sym, 'Q_Iasym' : model.numQ_asym,
        'Q_Isym': model.numQ_sym, 'R':model.numR,
        'Q_R' : model.numQ_R})

    df_sir['F'] = df_sir['F'] * SurreyPop / N
    df_sir['H'] = df_sir['H'] * SurreyPop / N
    df_sir['Isym'] = df_sir['Isym'] * SurreyPop / N
    df_sir['Iasym'] = df_sir['Iasym'] * SurreyPop / N
    df_sir['Q_Iasym'] = df_sir['Q_Iasym'] * SurreyPop / N
    df_sir['Q_Isym'] = df_sir['Q_Isym'] * SurreyPop / N
    df_sir['R'] = df_sir['R'] * SurreyPop / N
    df_sir['Q_R'] = df_sir['Q_R'] * SurreyPop / N
    
    df_sir.to_csv("./Forecasts_ExtremeHigh2/Parse{}c.csv".format(Num), index = False)

if __name__ == "__main__":
    N = 75000
    SurreyPop = 1160000

    cases_df = pd.read_csv("./CasesGovUK.csv")

    INIT_PreSym = cases_df.iloc[0:6]
    INIT_PreSym = (sum(INIT_PreSym['Cases']) / SurreyPop) * N    

    INIT_INFECTIOUS = cases_df.iloc[0:9]
    INIT_INFECTIOUS = (sum(INIT_INFECTIOUS['Cases']) / SurreyPop) * N
    INIT_Iasym = INIT_INFECTIOUS * 0.55
    INIT_Isym = INIT_INFECTIOUS - INIT_Iasym

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

    ALPHA = []
    for z in range(len(individual_ageGroups)):
        if individual_ageGroups[z] == '0-9':
            ALPHA.append(0.82)
        elif individual_ageGroups[z] == '10-19':
            ALPHA.append(0.77)
        elif individual_ageGroups[z] == '20-29':
            ALPHA.append(0.75)
        elif individual_ageGroups[z] == '30-39':
            ALPHA.append(0.75)
        elif individual_ageGroups[z] == '40-49':
            ALPHA.append(0.78)
        elif individual_ageGroups[z] == '50-59':
            ALPHA.append(0.87)
        elif individual_ageGroups[z] == '60-69':
            ALPHA.append(0.78)
        elif individual_ageGroups[z] == '70-79':
            ALPHA.append(0.75)
        else:
            ALPHA.append(0.81)

    ageGroup_pctHospitalized = {'0-9': 0.0095,
                                '10-19': 0.0006,
                                '20-29': 0.02,
                                '30-39': 0.037,
                                '40-49': 0.0425,
                                '50-59': 0.117,
                                '60-69': 0.143,
                                '70-79': 0.215,
                                '80+': 0.3875}
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

    Repeats = 6
    for i in range(0, Repeats):
        SEIR_Model(Num=i)
