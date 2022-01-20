import numpy as np
import time
import pandas as pd
import os


# 每个等级的信号强度峰值差标准，均值和标准差
SI_mean = np.array([121.97, 95.34, 72.34, 44.63, 20.60])
SI_std = np.array([9.96, 7.20, 7.81, 8.49, 9.28])

# 每个位置和性别的健康椎间盘SI参数，均值和标准差，位置从L1\L2开始
SI_health_mean = np.array([99.86, 96.98, 94.15, 93.21, 99.70])
SI_health_std = np.array([12.13, 10.28, 7.91, 9.05, 9.92])

# 每个位置和性别的健康椎间盘DH参数，均值和标准差，位置从L1\L2开始
DH_male_mean = np.array([13.15, 14.86, 16.15, 17.10, 17.11])
DH_male_std = np.array([1.28, 1.45, 1.47, 1.65, 2.24])

DH_female_mean = np.array([14.10, 16.37, 17.51, 18.42, 17.65])
DH_female_std = np.array([1.53, 1.63, 1.83, 1.91, 2.39])


# 每个位置和性别的健康椎间盘DHI参数，均值和标准差
DHI_male_mean = np.array([0.2768, 0.3083, 0.3398, 0.3688, 0.3715])
DHI_male_std = np.array([0.0351, 0.0386, 0.0413, 0.0465, 0.0648])

DHI_female_mean = np.array([0.2919, 0.3292, 0.3573, 0.3841, 0.3705])
DHI_female_std = np.array([0.0392, 0.0408, 0.0484, 0.0497, 0.0622])


# 每个位置和性别的健康椎间盘DWR参数，均值和标准差
DWR_male_mean = np.array([0.2164, 0.2315, 0.2468, 0.2572, 0.2707])
DWR_male_std = np.array([0.0222, 0.0239, 0.0277, 0.0311, 0.0369])

DWR_female_mean = np.array([0.2119, 0.2283, 0.2406, 0.2490, 0.2545])
DWR_female_std = np.array([0.0237, 0.0248, 0.0268, 0.0283, 0.0318])


def signal_fenji(signal_value):
    each_SI = signal_value
    if each_SI > SI_mean[0]:
        each_SI_grade = 1
    elif each_SI < SI_mean[4]:
        each_SI_grade = 5
    else:
        deta_each_SI = np.abs(each_SI - SI_mean)
        deta_each_SI = deta_each_SI.tolist()
        first_min = deta_each_SI.index(min(deta_each_SI))
        temp_deta_each_SI = deta_each_SI
        temp_deta_each_SI[first_min] = 1000
        seco_min = temp_deta_each_SI.index(min(temp_deta_each_SI))

        distance = np.abs(each_SI - SI_mean[first_min]) / SI_std[first_min] - np.abs(each_SI - SI_mean[seco_min]) / SI_std[seco_min]

        if distance > 0:
            each_SI_grade = seco_min + 1
        else:
            each_SI_grade = first_min + 1

    return each_SI_grade


def SI_quantitative(SI_input):
    taxian_per = (SI_input - SI_health_mean) / SI_health_std
    return taxian_per


def DH_quantitative(DH_input, sex_input):
    if sex_input == 0:
        taxian_per = -(1 - DH_input/DH_male_mean)
    else:
        taxian_per = -(1 - DH_input / DH_female_mean)

    return taxian_per


def DHI_quantitative(DHI_input, sex_input):
    if sex_input == 0:
        taxian_per = (DHI_input-DHI_male_mean)/DHI_male_std
    else:
        taxian_per = (DHI_input-DHI_female_mean)/DHI_female_std

    return taxian_per


def DWR_quantitative(DWR_input, sex_input):
    if sex_input == 0:
        taxian_per = (DWR_input-DWR_male_mean)/DWR_male_std
    else:
        taxian_per = (DWR_input-DWR_female_mean)/DWR_female_std

    return taxian_per


def quantitative_analysis(disc_si_dif_final, HD, DHI, DWR, sex_input):
    quantitative_results = []
    # 信号强度定性分级
    SI_grade = []
    for i in range(5):
        each_lumbar_level_grade = signal_fenji(disc_si_dif_final[i])
        SI_grade.append(each_lumbar_level_grade)
    quantitative_results.append(SI_grade)
    # SI定量分析
    SI_per = SI_quantitative(disc_si_dif_final)
    quantitative_results.append(SI_per)
    # DH定量分析
    DH_per = DH_quantitative(HD, sex_input)
    quantitative_results.append(DH_per)
    # DHI定量分析
    DHI_per = DHI_quantitative(DHI, sex_input)
    quantitative_results.append(DHI_per)
    # DWR定量分析
    DWR_per = DWR_quantitative(DWR, sex_input)
    quantitative_results.append(DWR_per)

    return quantitative_results
