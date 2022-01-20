import numpy as np
import cv2
import math
from collections import Counter
from scipy import signal


def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
    gaussKenrnel_x = cv2.getGaussianKernel(sigma, W, cv2.CV_64F)
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    gaussBlur_x = signal.convolve2d(image, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    gaussKenrnel_y = cv2.getGaussianKernel(sigma, H, cv2.CV_64F)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy


def get_pixel_value(inputs_Sigs, output_Sigs):
    inputs_Sigs = cv2.cvtColor(inputs_Sigs, cv2.COLOR_BGR2GRAY)
    blurImage = np.round(inputs_Sigs)
    inputs_Sigs = blurImage.astype(np.uint8)
    pixel_value_all = []
    for i in range(math.ceil(len(output_Sigs)/2)):
        pixel_weizhi = np.where(output_Sigs[2 * i + 1, :, :] == 1)
        pixel_weizhi_w = pixel_weizhi[1]
        pixel_weizhi_h = pixel_weizhi[0]
        pixel_value_slice = []
        for j in range(len(pixel_weizhi_w)):
            pixel_value_point = inputs_Sigs[pixel_weizhi_h[j], pixel_weizhi_w[j]]
            pixel_value_slice.append(pixel_value_point)
        pixel_value_all.append(pixel_value_slice)
    return pixel_value_all


def calcu_Sigs(inputs, output_Sigs):
    pixel_value_all = get_pixel_value(inputs, output_Sigs)
    firstpeak = []
    secondpeak = []
    most_pixel = []
    for i in range(len(pixel_value_all)):
        most_pixel_slice = Counter(pixel_value_all[i]).most_common(1)[0][0]
        most_pixel.append(most_pixel_slice)
        hist = cv2.calcHist([np.mat(pixel_value_all[i])], [0], None, [256], [0.0, 255.0])
        maxLoc = np.where(hist == np.max(hist))
        firstPeak_slice = maxLoc[0][0]
        firstpeak.append(firstPeak_slice)

        measureDists = np.zeros([256], np.float32)
        for k in range(256):
            measureDists[k] = math.fabs(k - firstPeak_slice * 0.95) * hist[k]  # 原始的系数是1.15
        maxLoc2 = np.where(measureDists == np.max(measureDists))
        secondPeak_slice = maxLoc2[0][0]
        secondpeak.append(secondPeak_slice)

    # 分别计算五个腰椎间盘高低信号强度（两个波峰）的差值，还有脑脊液和髂骨前脂肪区域的信号强度
    most_pixel_array = np.array(most_pixel, dtype=int)
    secondpeak_array = np.array(secondpeak, dtype=int)
    diff_fir_sec_peaks = np.abs((most_pixel_array - secondpeak_array))
    diff_fir_sec_peaks[0] = most_pixel_array[0]
    diff_fir_sec_peaks[6] = most_pixel_array[6]

    disc_si_dif_final = [diff_fir_sec_peaks[1], diff_fir_sec_peaks[2], diff_fir_sec_peaks[3],
                   diff_fir_sec_peaks[4], diff_fir_sec_peaks[5]]

    SI_disc_big = []
    diff_peaks = most_pixel_array - secondpeak_array
    for j in range(len(diff_peaks)):
        if diff_peaks[j] > 0:
            SI_disc = most_pixel_array[j]
        else:
            SI_disc = secondpeak_array[j]
        SI_disc_big.append(SI_disc)
    SI_big_final = [SI_disc_big[1], SI_disc_big[2], SI_disc_big[3], SI_disc_big[4], SI_disc_big[5],
                         diff_fir_sec_peaks[0], diff_fir_sec_peaks[6]]

    disc_si_dif_final = np.array(disc_si_dif_final)*255/int(diff_fir_sec_peaks[0])
    return SI_big_final, disc_si_dif_final