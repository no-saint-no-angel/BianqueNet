import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

"""
    这个文件的功能是：作为一个函数，提供在标准上制作所有散点图的功能。
    包括，（1）在不同椎间盘结构位置上，制作SI，DH,DHI,HDR标准图谱上的散点图
    （2）在不同的椎间盘退变等级标准谱图上，制作每个结构位置的SI散点图
"""
base_dir = './input/baseline_range'

"""
    保证版本号不能太新，否则报错。安装低版本：pip3 install xlrd==1.2.0
"""


def scatter_mean_std(si_dj_path, sex, age, SI, DH, DHI, HDR):
    if not os.path.exists(si_dj_path):
        os.makedirs(si_dj_path)
    # SI的等级
    data_SI_excel_name = os.path.join(os.path.join(base_dir, 'SI'), 'SI' + '.xlsx')
    # SI数据读取
    SI_biaozhun = pd.read_excel(data_SI_excel_name, 'SI_trend', usecols=[0, 1, 2])

    # SI的位置
    SI_dir = os.path.join(base_dir, 'SI')
    data_SI_L1L2_excel_name = os.path.join(SI_dir, 'SI_L1L2_trend' + '.xlsx')
    data_SI_L2L3_excel_name = os.path.join(SI_dir, 'SI_L2L3_trend' + '.xlsx')
    data_SI_L3L4_excel_name = os.path.join(SI_dir, 'SI_L3L4_trend' + '.xlsx')
    data_SI_L4L5_excel_name = os.path.join(SI_dir, 'SI_L4L5_trend' + '.xlsx')
    data_SI_L5S1_excel_name = os.path.join(SI_dir, 'SI_L5S1_trend' + '.xlsx')
    # SI数据读取
    SI_L1L2 = pd.read_excel(data_SI_L1L2_excel_name, 'SI trend', usecols=[0, 1])
    SI_L2L3 = pd.read_excel(data_SI_L2L3_excel_name, 'SI trend', usecols=[0, 1])
    SI_L3L4 = pd.read_excel(data_SI_L3L4_excel_name, 'SI trend', usecols=[0, 1])
    SI_L4L5 = pd.read_excel(data_SI_L4L5_excel_name, 'SI trend', usecols=[0, 1])
    SI_L5S1 = pd.read_excel(data_SI_L5S1_excel_name, 'SI trend', usecols=[0, 1])

    # DH的位置
    DH_dir_base = os.path.join(base_dir, 'DH')
    DH_dir = os.path.join(DH_dir_base, str(sex))
    data_DH_L1L2_excel_name = os.path.join(DH_dir, 'DH_L1L2_trend' + '.xlsx')
    data_DH_L2L3_excel_name = os.path.join(DH_dir, 'DH_L2L3_trend' + '.xlsx')
    data_DH_L3L4_excel_name = os.path.join(DH_dir, 'DH_L3L4_trend' + '.xlsx')
    data_DH_L4L5_excel_name = os.path.join(DH_dir, 'DH_L4L5_trend' + '.xlsx')
    data_DH_L5S1_excel_name = os.path.join(DH_dir, 'DH_L5S1_trend' + '.xlsx')
    # DHI数据读取
    DH_L1L2 = pd.read_excel(data_DH_L1L2_excel_name, 'DH trend', usecols=[1, 2, 3], header=1)
    DH_L2L3 = pd.read_excel(data_DH_L2L3_excel_name, 'DH trend', usecols=[1, 2, 3], header=1)
    DH_L3L4 = pd.read_excel(data_DH_L3L4_excel_name, 'DH trend', usecols=[1, 2, 3], header=1)
    DH_L4L5 = pd.read_excel(data_DH_L4L5_excel_name, 'DH trend', usecols=[1, 2, 3], header=1)
    DH_L5S1 = pd.read_excel(data_DH_L5S1_excel_name, 'DH trend', usecols=[1, 2, 3], header=1)

    # DHI的位置
    DHI_dir_base = os.path.join(base_dir, 'DHI')
    DHI_dir = os.path.join(DHI_dir_base, str(sex))
    data_DHI_L1L2_excel_name = os.path.join(DHI_dir, 'DHI_L1L2_trend' + '.xlsx')
    data_DHI_L2L3_excel_name = os.path.join(DHI_dir, 'DHI_L2L3_trend' + '.xlsx')
    data_DHI_L3L4_excel_name = os.path.join(DHI_dir, 'DHI_L3L4_trend' + '.xlsx')
    data_DHI_L4L5_excel_name = os.path.join(DHI_dir, 'DHI_L4L5_trend' + '.xlsx')
    data_DHI_L5S1_excel_name = os.path.join(DHI_dir, 'DHI_L5S1_trend' + '.xlsx')
    # DHI数据读取
    DHI_L1L2 = pd.read_excel(data_DHI_L1L2_excel_name, 'DHI trend', usecols=[1, 2, 3], header=1)
    DHI_L2L3 = pd.read_excel(data_DHI_L2L3_excel_name, 'DHI trend', usecols=[1, 2, 3], header=1)
    DHI_L3L4 = pd.read_excel(data_DHI_L3L4_excel_name, 'DHI trend', usecols=[1, 2, 3], header=1)
    DHI_L4L5 = pd.read_excel(data_DHI_L4L5_excel_name, 'DHI trend', usecols=[1, 2, 3], header=1)
    DHI_L5S1 = pd.read_excel(data_DHI_L5S1_excel_name, 'DHI trend', usecols=[1, 2, 3], header=1)

    # HDR位置
    HDR_name = 'DWR'
    HDR_dir_base = os.path.join(base_dir, HDR_name)
    HDR_dir = os.path.join(HDR_dir_base, str(sex))
    data_HDR_L1L2_excel_name = os.path.join(HDR_dir, HDR_name + '_L1L2_trend' + '.xlsx')
    data_HDR_L2L3_excel_name = os.path.join(HDR_dir, HDR_name + '_L2L3_trend' + '.xlsx')
    data_HDR_L3L4_excel_name = os.path.join(HDR_dir, HDR_name + '_L3L4_trend' + '.xlsx')
    data_HDR_L4L5_excel_name = os.path.join(HDR_dir, HDR_name + '_L4L5_trend' + '.xlsx')
    data_HDR_L5S1_excel_name = os.path.join(HDR_dir, HDR_name + '_L5S1_trend' + '.xlsx')
    # HDR数据读取
    HDR_L1L2 = pd.read_excel(data_HDR_L1L2_excel_name, HDR_name + ' trend', usecols=[1, 2, 3], header=1)
    HDR_L2L3 = pd.read_excel(data_HDR_L2L3_excel_name, HDR_name + ' trend', usecols=[1, 2, 3], header=1)
    HDR_L3L4 = pd.read_excel(data_HDR_L3L4_excel_name, HDR_name + ' trend', usecols=[1, 2, 3], header=1)
    HDR_L4L5 = pd.read_excel(data_HDR_L4L5_excel_name, HDR_name + ' trend', usecols=[1, 2, 3], header=1)
    HDR_L5S1 = pd.read_excel(data_HDR_L5S1_excel_name, HDR_name + ' trend', usecols=[1, 2, 3], header=1)

    # 标准颜色设置,男性橙色，女性蓝色
    # 散点颜色设置
    c_point = sns.xkcd_palette(['black'])
    if sex == 1:
        c = sns.xkcd_palette(['orangered'])
    else:
        c = sns.xkcd_palette(['blue'])
    # 散点形状
    marker = 'd'
    # 散点横坐标设置，年龄
    point_age = 5*(age-20)/70
    # 散点纵坐标是各个椎间盘位置的值
    fig_size = (8, 12)

    # SI在每个椎间盘结构位置的上的散点图++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sns.set_context("talk", font_scale=1, rc={'line.linrwidth': 0.5})
    plt.rc('font', family='Times New Roman')
    f_SI = plt.figure(figsize=fig_size)
    # f_DHI.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace=0.3, hspace=0)  # 调整子图间距

    # L1L2
    f_SI.add_subplot(511)
    plt.title("SI")
    ax111 = sns.lineplot(x="age", y="L1~L2", palette=c, data=SI_L1L2, ci="sd")
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.8), ncol=1)
    ax111.scatter(point_age, SI[0], s=80, marker=marker, c=c_point)
    ax111.legend_.remove()
    # L2L3
    f_SI.add_subplot(512)
    ax112 = sns.lineplot(x="age", y="L2~L3", palette=c, data=SI_L2L3, ci="sd")
    ax112.scatter(point_age, SI[1], s=80, marker=marker, c=c_point)

    # L3L4
    f_SI.add_subplot(513)
    ax113 = sns.lineplot(x="age", y="L3~L4", palette=c, data=SI_L3L4, ci="sd")
    ax113.scatter(point_age, SI[2], s=80, marker=marker, c=c_point)

    # L4L5
    f_SI.add_subplot(514)
    ax114 = sns.lineplot(x="age", y="L4~L5", palette=c, data=SI_L4L5, ci="sd")
    ax114.scatter(point_age, SI[3], s=80, marker=marker, c=c_point)

    # L5S1
    f_SI.add_subplot(515)
    ax115 = sns.lineplot(x="age", y="L5~S1", palette=c, data=SI_L5S1, ci="sd")
    ax115.scatter(point_age, SI[4], s=80, marker=marker, c=c_point)
    plt.savefig(os.path.join(si_dj_path, "SI_weizhi.png"), bbox_inches='tight')

    # 在每个等级处画散点++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    sns.set_context("talk", font_scale=1, rc={'line.linrwidth': 0.5})
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(11.25, 12))
    ax1 = sns.lineplot(x="location", y="SI", hue="grade", data=SI_biaozhun, ci="sd")
    ax1.scatter(0, SI[0], s=80, marker=marker, c=c_point)
    ax1.scatter(1, SI[1], s=80, marker=marker, c=c_point)
    ax1.scatter(2, SI[2], s=80, marker=marker, c=c_point)
    ax1.scatter(3, SI[3], s=80, marker=marker, c=c_point)
    ax1.scatter(4, SI[4], s=80, marker=marker, c=c_point)
    plt.legend(loc='center right', bbox_to_anchor=(0.98, 0.855), ncol=1)
    plt.savefig(os.path.join(si_dj_path, "SI_dj.png"), bbox_inches='tight')
    # plt.show()

    # DH++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sns.set_context("talk", font_scale=1, rc={'line.linrwidth': 0.5})
    plt.rc('font', family='Times New Roman')
    f_DH = plt.figure(figsize=fig_size)
    plt.subplots_adjust(wspace=0.3, hspace=0)

    # L1L2
    f_DH.add_subplot(511)
    plt.title("DH")
    ax11 = sns.lineplot(x="age", y="L1~L2", hue="gender", palette=c, data=DH_L1L2, ci="sd")
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.8), ncol=1)
    ax11.scatter(point_age, DH[0], s=80, marker=marker, c=c_point)
    # L2L3
    f_DH.add_subplot(512)
    ax12 = sns.lineplot(x="age", y="L2~L3", hue="gender", palette=c, data=DH_L2L3, ci="sd")
    ax12.scatter(point_age, DH[1], s=80, marker=marker, c=c_point)
    ax12.legend_.remove()
    # L3L4
    f_DH.add_subplot(513)
    ax13 = sns.lineplot(x="age", y="L3~L4", hue="gender", palette=c, data=DH_L3L4, ci="sd")
    ax13.scatter(point_age, DH[2], s=80, marker=marker, c=c_point)
    ax13.legend_.remove()
    # L4L5
    f_DH.add_subplot(514)
    ax14 = sns.lineplot(x="age", y="L4~L5", hue="gender", palette=c, data=DH_L4L5, ci="sd")
    ax14.scatter(point_age, DH[3], s=80, marker=marker, c=c_point)
    ax14.legend_.remove()
    # L5S1
    f_DH.add_subplot(515)
    ax15 = sns.lineplot(x="age", y="L5~S1", hue="gender", palette=c, data=DH_L5S1, ci="sd")
    ax15.scatter(point_age, DH[4], s=80, marker=marker, c=c_point)
    ax15.legend_.remove()

    plt.savefig(os.path.join(si_dj_path, "DH.png"), bbox_inches='tight')

    # DHI++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sns.set_context("talk", font_scale=1, rc={'line.linrwidth': 0.5})
    plt.rc('font', family='Times New Roman')
    f_DHI = plt.figure(figsize=fig_size)
    plt.subplots_adjust(wspace=0.3, hspace=0)

    # L1L2
    f_DHI.add_subplot(511)
    plt.title("DHI")
    ax1 = sns.lineplot(x="age", y="L1~L2",hue="gender", palette=c, data=DHI_L1L2, ci="sd")
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.8), ncol=1)
    ax1.scatter(point_age, DHI[0], s=80, marker=marker, c=c_point)
    # L2L3
    f_DHI.add_subplot(512)
    ax2 = sns.lineplot(x="age", y="L2~L3",hue="gender", palette=c, data=DHI_L2L3, ci="sd")
    ax2.scatter(point_age, DHI[1], s=80, marker=marker, c=c_point)
    ax2.legend_.remove()
    # L3L4
    f_DHI.add_subplot(513)
    ax3 = sns.lineplot(x="age", y="L3~L4",hue="gender", palette=c, data=DHI_L3L4, ci="sd")
    ax3.scatter(point_age, DHI[2], s=80, marker=marker, c=c_point)
    ax3.legend_.remove()
    # L4L5
    f_DHI.add_subplot(514)
    ax4 = sns.lineplot(x="age", y="L4~L5",hue="gender", palette=c, data=DHI_L4L5, ci="sd")
    ax4.scatter(point_age, DHI[3], s=80, marker=marker, c=c_point)
    ax4.legend_.remove()
    # L5S1
    f_DHI.add_subplot(515)
    ax5 = sns.lineplot(x="age", y="L5~S1",hue="gender", palette=c, data=DHI_L5S1, ci="sd")
    ax5.scatter(point_age, DHI[4], s=80, marker=marker, c=c_point)
    ax5.legend_.remove()

    plt.savefig(os.path.join(si_dj_path, "DHI.png"), bbox_inches='tight')

    # HDR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sns.set_context("talk", font_scale=1, rc={'line.linrwidth': 0.5})
    plt.rc('font', family='Times New Roman')
    f_HDR = plt.figure(figsize=fig_size)
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # L1L2
    f_HDR.add_subplot(511)
    plt.title("HDR")
    ax6 = sns.lineplot(x="age", y="L1~L2",hue="gender", palette=c, data=HDR_L1L2, ci="sd")
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.8), ncol=1)
    ax6.scatter(point_age, HDR[0], s=80, marker=marker, c=c_point)
    # L2L3
    f_HDR.add_subplot(512)
    ax7 = sns.lineplot(x="age", y="L2~L3",hue="gender", palette=c, data=HDR_L2L3, ci="sd")
    ax7.scatter(point_age, HDR[1], s=80, marker=marker, c=c_point)
    ax7.legend_.remove()
    # L3L4
    f_HDR.add_subplot(513)
    ax8 = sns.lineplot(x="age", y="L3~L4",hue="gender", palette=c, data=HDR_L3L4, ci="sd")
    ax8.scatter(point_age, HDR[2], s=80, marker=marker, c=c_point)
    ax8.legend_.remove()
    # L4L5
    f_HDR.add_subplot(514)
    ax9 = sns.lineplot(x="age", y="L4~L5",hue="gender", palette=c, data=HDR_L4L5, ci="sd")
    ax9.scatter(point_age, HDR[3], s=80, marker=marker, c=c_point)
    ax9.legend_.remove()
    # L5S1
    f_HDR.add_subplot(515)
    ax10 = sns.lineplot(x="age", y="L5~S1",hue="gender", palette=c, data=HDR_L5S1, ci="sd")
    ax10.scatter(point_age, HDR[4], s=80, marker=marker, c=c_point)
    ax10.legend_.remove()

    plt.savefig(os.path.join(si_dj_path, "HDR.png"), bbox_inches='tight')