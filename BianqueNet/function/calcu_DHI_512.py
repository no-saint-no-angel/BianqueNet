import numpy as np
import cv2
import math


# VH ********************************************************************************
# L1-L5
def calcu_HV(L1_calcu_HV):
    src = L1_calcu_HV.copy()
    gray_harris = np.float32(src)
    gray_st = gray_harris.copy()
    # 2、Shi-Tomasi 角点检测
    maxCorners = 4
    qualityLevel = 0.01
    minDistance = 21
    block_size_s = 9
    k_s = 0.04
    corners_st = cv2.goodFeaturesToTrack(gray_st, maxCorners, qualityLevel, minDistance, corners=None, mask=None,
                                         blockSize=block_size_s, useHarrisDetector=None, k=k_s)

    # 第二种
    corners_label_final2 = np.int0(np.squeeze(corners_st))
    corners_label_final = corners_label_final2.copy()
    corners_label_zanshi = corners_label_final2.copy()
    corners_label_zanshi = corners_label_zanshi.tolist()
    # 因为这里得出的corners_label_final2的排序是混乱的，需要排序
    # 先找最大值和最小值对应的第4个（右下角）和第1点（左上角）
    sum_final2_wh = np.sum(corners_label_final, axis=1)
    wh_max = np.where(sum_final2_wh == np.max(sum_final2_wh))
    wh_min = np.where(sum_final2_wh == np.min(sum_final2_wh))
    wh_max = wh_max[0]
    wh_min = wh_min[0]
    # 根据wh_max和wh_min索引，将corners_label_final中的第4个和第1个位置，用corners_label_final2中的值替换
    corners_label_final[3, 0] = corners_label_final2[wh_max[0], 0]  # w
    corners_label_final[3, 1] = corners_label_final2[wh_max[0], 1]  # h
    corners_label_final[0, 0] = corners_label_final2[wh_min[0], 0]  # w
    corners_label_final[0, 1] = corners_label_final2[wh_min[0], 1]  # h
    # 再找剩下两个不是最值的索引，就是第2个（右上角）和第3个（左下角）的点，得到的是数组类型数据，转换成列表进行删除操作
    a_a = corners_label_final2[wh_max[0], :]
    b_b = corners_label_final2[wh_min[0], :]
    a_a = a_a.tolist()
    b_b = b_b.tolist()
    corners_label_zanshi.remove(a_a)
    corners_label_zanshi.remove(b_b)
    corners_label_zanshi = np.array(corners_label_zanshi)

    if corners_label_zanshi[0, 1] > corners_label_zanshi[1, 1]:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[0, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[1, 1]  # h
    else:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[1, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[0, 1]  # h
    corners_label_final_end = corners_label_final.copy()
    # 比较右上角的w值否小于左上角的，是的话，调换1，2位置，否则不调换
    if corners_label_final[1, 0] < corners_label_final[0, 0]:
        corners_label_final_end[0, 0] = corners_label_final[1, 0]
        corners_label_final_end[0, 1] = corners_label_final[1, 1]
        corners_label_final_end[1, 0] = corners_label_final[0, 0]
        corners_label_final_end[1, 1] = corners_label_final[0, 1]
    # 比较右下角的w值否小于左下角的，是的话，调换3，4位置，否则不调换
    if corners_label_final[3, 0] < corners_label_final[2, 0]:
        corners_label_final_end[2, 0] = corners_label_final[3, 0]
        corners_label_final_end[2, 1] = corners_label_final[3, 1]
        corners_label_final_end[3, 0] = corners_label_final[2, 0]
        corners_label_final_end[3, 1] = corners_label_final[2, 1]
    # 提取角点位置
    point_L_h_bf_test = corners_label_final_end[:, 1]
    point_L_w_bf_test = corners_label_final_end[:, 0]

    if len(point_L_h_bf_test) == 4:
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    else:
        # 打印角点数，起提示作用
        print('The number of corners of the detected cone is', len(point_L_h_bf_test))
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    # 计算椎体的宽度
    L_wid_up = math.hypot((point_L_h[0] - point_L_h[1]), (point_L_w[0] - point_L_w[1]))
    L_wid_down = math.hypot((point_L_h[2] - point_L_h[3]), (point_L_w[2] - point_L_w[3]))
    L_wid = (L_wid_up + L_wid_down) / 2.0
    HV = np.sum(src) / L_wid
    return HV, point_L_h, point_L_w


# S1
def calcu_HV_S1(L1_calcu_HV):
    # 保存原始L1，计算像素个数和
    src = L1_calcu_HV.copy()
    # 对椎体进行角点检测
    gray_harris = np.float32(src)
    gray_st = gray_harris.copy()

    # 2、Shi-Tomasi 角点检测
    maxCorners = 4
    qualityLevel = 0.01
    minDistance = 21
    block_size_s = 9
    k_s = 0.04
    corners_st = cv2.goodFeaturesToTrack(gray_st, maxCorners, qualityLevel, minDistance, corners=None, mask=None,
                                         blockSize=block_size_s, useHarrisDetector=None, k=k_s)
    # 第二种
    corners_label_final2 = np.int0(np.squeeze(corners_st))
    corners_label_final = corners_label_final2.copy()
    corners_label_zanshi = corners_label_final2.copy()
    corners_label_zanshi = corners_label_zanshi.tolist()
    # 先找最大值和最小值对应的第4个（右下角）和第1点（左上角）
    sum_final2_wh = np.sum(corners_label_final, axis=1)
    wh_max = np.where(sum_final2_wh == np.max(sum_final2_wh))
    wh_min = np.where(sum_final2_wh == np.min(sum_final2_wh))
    wh_max = wh_max[0]
    wh_min = wh_min[0]
    # 根据wh_max和wh_min索引，将corners_label_final中的第4个和第1个位置，用corners_label_final2中的值替换
    corners_label_final[3, 0] = corners_label_final2[wh_max[0], 0]  # w
    corners_label_final[3, 1] = corners_label_final2[wh_max[0], 1]  # h
    corners_label_final[0, 0] = corners_label_final2[wh_min[0], 0]  # w
    corners_label_final[0, 1] = corners_label_final2[wh_min[0], 1]  # h
    # 再找剩下两个不是最值的索引，就是第2个（右上角）和第3个（左下角）的点，得到的是数组类型数据，转换成列表进行删除操作
    a_a = corners_label_final2[wh_max[0], :]
    b_b = corners_label_final2[wh_min[0], :]
    a_a = a_a.tolist()
    b_b = b_b.tolist()
    corners_label_zanshi.remove(a_a)
    corners_label_zanshi.remove(b_b)
    corners_label_zanshi = np.array(corners_label_zanshi)

    if (corners_label_zanshi[0,0]+corners_label_zanshi[0,1]) > (corners_label_zanshi[1,0]+corners_label_zanshi[1,1]):
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[0, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[1, 1]  # h
    else:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[1, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[0, 1]  # h

    corners_label_final_end = corners_label_final.copy()
    # 比较右上角的w值否小于左上角的，是的话，调换1，2位置，否则不调换
    if corners_label_final[1, 0] < corners_label_final[0, 0]:
        corners_label_final_end[0, 0] = corners_label_final[1, 0]
        corners_label_final_end[0, 1] = corners_label_final[1, 1]
        corners_label_final_end[1, 0] = corners_label_final[0, 0]
        corners_label_final_end[1, 1] = corners_label_final[0, 1]
    # 比较右下角的w值否小于左下角的，是的话，调换3，4位置，否则不调换
    if corners_label_final[3, 0] < corners_label_final[2, 0]:
        corners_label_final_end[2, 0] = corners_label_final[3, 0]
        corners_label_final_end[2, 1] = corners_label_final[3, 1]
        corners_label_final_end[3, 0] = corners_label_final[2, 0]
        corners_label_final_end[3, 1] = corners_label_final[2, 1]
    # 提取角点位置
    point_L_h_bf_test = corners_label_final_end[:, 1]
    point_L_w_bf_test = corners_label_final_end[:, 0]
    if len(point_L_h_bf_test) == 4:
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    else:
        # 打印角点数，起提示作用
        print('The number of corners of the detected cone is', len(point_L_h_bf_test))
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()

    # 现在得到四个点，计算椎体的宽度
    L_wid_up = math.hypot((point_L_h[0] - point_L_h[1]), (point_L_w[0] - point_L_w[1]))
    L_wid_down = math.hypot((point_L_h[2] - point_L_h[3]), (point_L_w[2] - point_L_w[3]))
    L_wid = (L_wid_up + L_wid_down) / 2.0
    HV = np.sum(src) / L_wid
    return HV, point_L_h, point_L_w


# WD ********************************************************************************
# 计算完整的腰椎间盘的宽度big_WD
def calcu_big_WD(L1_L2_disc_calcu_HD, point_D12_1, point_D12_2):
    # 对索引值进行取整，
    h0 = int(point_D12_1[0])
    w0 = int(point_D12_1[1])
    h1 = int(point_D12_2[0])
    w1 = int(point_D12_2[1])
    L1_L2_disc_calcu_HD = np.array(L1_L2_disc_calcu_HD)
    len_pic = len(L1_L2_disc_calcu_HD)
    if h0 == h1:
        print('斜率为零！')
        # 计算最右边的点,从右边中点开始往右遍历，记录像素值为0时，左边一个点
        for s in range(w1, len_pic):
            if L1_L2_disc_calcu_HD[h1, s] == 0:
                point_you = [h1, s - 1]
                break

        # 计算最左边的点，从左边点开始往左遍历，记录像素值为0时，右边一个点
        for t in range(w0, 0, -1):
            if L1_L2_disc_calcu_HD[h0, t] == 0:
                point_zuo = [h0, t + 1]
                break

    else:
        m, b = slope(point_D12_1[0], point_D12_1[1], point_D12_2[0], point_D12_2[1])  # 计算斜率还是用没有取整的数，为了精确
        # 计算最右边的点,从右边中点开始往右遍历，记录像素值为0时，左边一个点
        for s in range(w1, len_pic):
            if L1_L2_disc_calcu_HD[int(m * s + b), s] == 0:
                point_you = [int(m * (s - 1) + b), s - 1]
                break

        # 计算最左边的点，从左边点开始往左遍历，记录像素值为0时，右边一个点
        for t in range(w0, 0, -1):
            if L1_L2_disc_calcu_HD[int(m * t + b), t] == 0:
                point_zuo = [int(m * (t + 1) + b), t + 1]
                break

    point_you_zuo = np.array(point_you) - np.array(point_zuo)
    big_WD = math.hypot(point_you_zuo[0], point_you_zuo[1])

    return big_WD


# calcu_WD计算腰椎间盘宽度，腰椎间盘80%区域四个分割点
def calcu_WD(L1_L2_disc_calcu_HD, point_L1_h, point_L1_w, point_L2_h, point_L2_w):
    point_fenge = []
    # 上一个椎体的下面两个顶点，下一椎体的上面两个顶点
    point_L1_3 = np.array([point_L1_h[2], point_L1_w[2]])
    point_L1_4 = np.array([point_L1_h[3], point_L1_w[3]])
    point_L2_1 = np.array([point_L2_h[0], point_L2_w[0]])
    point_L2_2 = np.array([point_L2_h[1], point_L2_w[1]])
    # 腰椎间盘前后中点，point_Di（i+1）_k，k取1,2
    point_D12_1 = (point_L1_3 + point_L2_1) / 2
    point_D12_2 = (point_L1_4 + point_L2_2) / 2
    # 腰椎间盘宽度small_WD，small_WD，是根据椎体的四个角点来确定腰椎间盘的范围，进而计算的。在这四个角点得出的分割线之外还有膨出的腰椎间盘，
    # 故这个不是真正意义上的腰椎间盘宽度，所以称为small_WD,包括膨出的腰椎间盘的宽度称为big_WD。
    point_D12_21 = point_D12_2 - point_D12_1
    small_WD = math.hypot(point_D12_21[0], point_D12_21[1])
    big_WD = calcu_big_WD(L1_L2_disc_calcu_HD, point_D12_1, point_D12_2)
    # 计算前后中点h,w方向差值
    delta_12_h = math.fabs(point_D12_1[0] - point_D12_2[0])
    delta_12_w = math.fabs(point_D12_1[1] - point_D12_2[1])
    # 椎间盘中心点point_D12_c0
    point_D12_c0 = (point_D12_1 + point_D12_2) / 2
    # 计算腰椎间盘80%区域四个分割点,分割占比miu=0.8，分割边界的高度qiang=1，表示一个单位的delta_12_w
    miu = 0.8
    qiang = 0.75  # 0.75
    miu_half = 0.5*miu
    qiang_half = 0.5*qiang
    # 在这里要分两种情况，一种是前后中点连线的斜率是负的，前中点高于后中点（h值小）：另一种是正的，前中点低于后中点（h值大）
    if point_D12_1[0] < point_D12_2[0]:  # 对比h值
        point_D12_c0lu = np.array([point_D12_c0[0] - miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0lu))
        point_D12_c0ld = np.array([point_D12_c0[0] - miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ld))
        point_D12_c0ru = np.array([point_D12_c0[0] + miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ru))
        point_D12_c0rd = np.array([point_D12_c0[0] + miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0rd))
    else:
        point_D12_c0lu = np.array([point_D12_c0[0] + miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0lu))
        point_D12_c0ld = np.array([point_D12_c0[0] + miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ld))
        point_D12_c0ru = np.array([point_D12_c0[0] - miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ru))
        point_D12_c0rd = np.array([point_D12_c0[0] - miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0rd))

    return small_WD, big_WD, point_fenge


# DH ********************************************************************************
# 计算斜率
def slope(h0, w0, h1, w1):
    m = (h1-h0)/(w1-w0)  # 把h当成y，w当成x,
    b = h0 - m*w0
    return m, b


# 计算两条分割线上的所有像素点位置
def get_pixel(h0, w0, h1, w1):
    piont_h = []
    piont_w = []
    if w0 == w1:
        print('斜率不存在！')
        for j in range(h0, h1):
            piont_h.append(j)
            piont_w.append(w0)
    else:
        m, b = slope(h0, w0, h1, w1)
        for j in range(h0, h1):
            point_w_temp = (j - b)/m
            piont_w.append(round(point_w_temp))
            piont_h.append(j)

    return piont_h, piont_w


# calcu_HD计算腰椎间盘高度
def calcu_HD(L1_L2_disc_calcu_HD, point_D12_l_up, point_D12_l_down, point_D12_r_up, point_D12_r_down, w_D):
    L1_L2_disc_HD = L1_L2_disc_calcu_HD.copy()
    len_pic = len(L1_L2_disc_HD)
    # 腰椎间盘80%区域分割点，四个
    point_D12_l_up = point_D12_l_up
    point_D12_l_down = point_D12_l_down
    point_D12_r_up = point_D12_r_up
    point_D12_r_down = point_D12_r_down
    # 腰椎间盘80%区域分割点连成的两条直线（前）上的所有像素点
    point_D12_l_up2down_h, point_D12_l_up2down_w = get_pixel(point_D12_l_up[0], point_D12_l_up[1], point_D12_l_down[0],
                                                             point_D12_l_down[1])
    len_D12_l_up2down_h = len(point_D12_l_up2down_h)
    # 从整个腰椎间盘取80%，去除前10%
    for j in range(len_D12_l_up2down_h):
        for i in range(point_D12_l_up2down_w[j]):
            L1_L2_disc_HD[point_D12_l_up2down_h[j], i] = 0
    point_D12_r_up2down_h, point_D12_r_up2down_w = get_pixel(point_D12_r_up[0], point_D12_r_up[1], point_D12_r_down[0],
                                                             point_D12_r_down[1])
    len_D12_r_up2down_h = len(point_D12_r_up2down_h)
    # 从整个腰椎间盘取80%，去除后10%
    for j in range(len_D12_r_up2down_h):
        for i in range(point_D12_r_up2down_w[j], len_pic):
            L1_L2_disc_HD[point_D12_r_up2down_h[j], i] = 0
    sum_A = np.sum(L1_L2_disc_HD)
    HD = sum_A/w_D
    return HD


# DHI&HDR ********************************************************************************
def calcu_DHI(output):
    # 先提取腰椎间盘上下两个椎体的各四个角点，point_Li_j，i从1到5，j从1到4
    # 腰椎间盘前后中点，point_Di（i+1）_k，k取1,2
    # 腰椎间盘80%区域前后中点，point_Di（i+1）_l，point_Di（i+1）_r
    # 腰椎间盘80%区域分割点，四个，point_Di（i+1）_l_up,point_Di（i+1）_l_down,point_Di（i+1）_r_up,point_Di（i+1）_r_down
    # 腰椎间盘80%区域分割点连成的两条直线（前后）上的所有像素点，point_Di（i+1）_l_up2down,point_Di（i+1）_r_up2down,加h后缀表示
    # 像素点的h方向的值组成的list
    # 腰椎间盘宽度w_D
    DHI_big = []
    DWR_big = []
    HV_big = []
    HD_big = []
    WD_big = []
    point_fenge_big = []

    # DHI12
    print('Calculating the DHI of the L1_L2_disc......')
    L1 = output[2, :, :]
    HV1, point_L1_h, point_L1_w = calcu_HV(L1)
    HV_big.append(HV1)
    L1_L2_disc = output[3, :, :]
    L2 = output[4, :, :]
    HV2, point_L2_h, point_L2_w = calcu_HV(L2)
    HV_big.append(HV2)
    small_WD12, big_WD12, point_fenge12 = calcu_WD(L1_L2_disc, point_L1_h, point_L1_w, point_L2_h, point_L2_w)
    WD_big.append(big_WD12)
    point_fenge_big.append(np.array(point_fenge12))
    DH12 = calcu_HD(L1_L2_disc, point_fenge12[0], point_fenge12[1], point_fenge12[2], point_fenge12[3], small_WD12)
    HD_big.append(DH12)
    # 计算DHI指数
    DHI12 = 2 * DH12 / (HV1 + HV2)
    DHI_big.append(DHI12)
    print('The DHI of the L1_L2_disc is ', DHI12)
    # 计算高宽比
    DWR12 = DH12 / big_WD12
    DWR_big.append(DWR12)
    print('The DWR of the L1_L2_disc is ', DWR12)

    # DHI23
    print('Calculating the DHI of the L2_L3_disc......')
    L2_L3_disc = output[5, :, :]
    L3 = output[6, :, :]
    HV3, point_L3_h, point_L3_w = calcu_HV(L3)
    HV_big.append(HV3)
    small_WD23, big_WD23, point_fenge23 = calcu_WD(L2_L3_disc, point_L2_h, point_L2_w, point_L3_h, point_L3_w)
    WD_big.append(big_WD23)
    point_fenge_big.append(np.array(point_fenge23))
    DH23 = calcu_HD(L2_L3_disc, point_fenge23[0], point_fenge23[1], point_fenge23[2], point_fenge23[3], small_WD23)
    HD_big.append(DH23)
    DHI23 = 2 * DH23 / (HV2 + HV3)
    DHI_big.append(DHI23)
    print('The DHI of the L2_L3_disc is ', DHI23)
    # 计算高宽比
    DWR23 = DH23 / big_WD23
    DWR_big.append(DWR23)
    print('The DWR of the L2_L3_disc is ', DWR23)

    # DHI34
    print('Calculating the DHI of the L3_L4_disc......')
    L3_L4_disc = output[7, :, :]
    L4 = output[8, :, :]
    HV4, point_L4_h, point_L4_w = calcu_HV(L4)
    HV_big.append(HV4)
    small_WD34, big_WD34, point_fenge34 = calcu_WD(L3_L4_disc, point_L3_h, point_L3_w, point_L4_h, point_L4_w)
    WD_big.append(big_WD34)
    point_fenge_big.append(np.array(point_fenge34))
    DH34 = calcu_HD(L3_L4_disc, point_fenge34[0], point_fenge34[1], point_fenge34[2], point_fenge34[3], small_WD34)
    HD_big.append(DH34)
    DHI34 = 2 * DH34 / (HV3 + HV4)
    DHI_big.append(DHI34)
    print('The DHI of the L3_L4_disc is ', DHI34)
    # 计算高宽比
    DWR34 = DH34 / big_WD34
    DWR_big.append(DWR34)
    print('The DWR of the L3_L4_disc is ', DWR34)

    # DHI45
    print('Calculating the DHI of the L4_L5_disc......')
    L4_L5_disc = output[9, :, :]
    L5 = output[10, :, :]
    HV5, point_L5_h, point_L5_w = calcu_HV(L5)
    HV_big.append(HV5)
    small_WD45, big_WD45, point_fenge45 = calcu_WD(L4_L5_disc, point_L4_h, point_L4_w, point_L5_h, point_L5_w)
    WD_big.append(big_WD45)
    point_fenge_big.append(np.array(point_fenge45))
    DH45 = calcu_HD(L4_L5_disc, point_fenge45[0], point_fenge45[1], point_fenge45[2], point_fenge45[3], small_WD45)
    HD_big.append(DH45)
    DHI45 = 2 * DH45 / (HV4 + HV5)
    DHI_big.append(DHI45)
    print('The DHI of the L4_L5_disc is ', DHI45)
    # 计算高宽比
    DWR45 = DH45 / big_WD45
    DWR_big.append(DWR45)
    print('The DWR of the L4_L5_disc is ', DWR45)


    # DHI5S1
    print('Calculating the DHI of the L5_S1_disc......')
    L5_S1_disc = output[11, :, :]
    S1 = output[12, :, :]
    HVS1, point_S1_h, point_S1_w = calcu_HV_S1(S1)
    HV_big.append(HVS1)
    small_WD5S1, big_WD5S1, point_fenge5S1 = calcu_WD(L5_S1_disc, point_L5_h, point_L5_w, point_S1_h, point_S1_w)
    WD_big.append(big_WD5S1)
    point_fenge_big.append(np.array(point_fenge5S1))
    DH5S1 = calcu_HD(L5_S1_disc, point_fenge5S1[0], point_fenge5S1[1], point_fenge5S1[2], point_fenge5S1[3], small_WD5S1)
    HD_big.append(DH5S1)
    # DHI5S1 = 2 * DH5S1 / (HVS1 + HV5)
    # 稍微更改一下第五节腰椎DHI指数的计算公式，不用S1的高度了
    DHI5S1 = DH5S1 / HV5
    DHI_big.append(DHI5S1)
    print('The DHI of the L5_S1_disc is ', DHI5S1)
    # 计算高宽比
    DWR5S1 = DH5S1 / big_WD5S1
    DWR_big.append(DWR5S1)
    print('The DWR of the L5_S1_disc is ', DWR5S1)

    # 存储所有角点
    point_big_h = [point_L1_h, point_L2_h, point_L3_h, point_L4_h, point_L5_h, point_S1_h]
    point_big_w = [point_L1_w, point_L2_w, point_L3_w, point_L4_w, point_L5_w, point_S1_w]
    # 储存所有分割点
    point_fenge_big = np.array(point_fenge_big)
    point_fenge_h_big = [point_fenge_big[0, :, 0], point_fenge_big[1, :, 0], point_fenge_big[2, :, 0],
                         point_fenge_big[3, :, 0], point_fenge_big[4, :, 0]]
    point_fenge_w_big = [point_fenge_big[0, :, 1], point_fenge_big[1, :, 1], point_fenge_big[2, :, 1],
                         point_fenge_big[3, :, 1], point_fenge_big[4, :, 1]]

    return DHI_big, DWR_big, HD_big, HV_big, point_big_h, point_big_w, point_fenge_h_big, point_fenge_w_big

