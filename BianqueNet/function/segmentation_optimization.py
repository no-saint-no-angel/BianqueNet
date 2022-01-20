import numpy as np
import cv2


def max_area_extract(input_chanel):
    chanel2 = np.array(input_chanel, dtype='uint8')
    contours, hierarchy = cv2.findContours(chanel2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        area_list = []
        img_contours = []
        img_area_chenal = []
        for j in range(len(contours)):
            area = cv2.contourArea(contours[j])
            area_list.append(area)
            img_temp = np.zeros(chanel2.shape, np.uint8)
            img_contours.append(img_temp)
            cv2.drawContours(img_contours[j], contours, j, (255, 255, 255), -1)
            img_area_chenal.append(img_contours[j])
        max_where = np.where(area_list == np.max(area_list, axis=0))
        max_area_chenal = max_where[0]
        img_area_chenal_max = img_area_chenal[max_area_chenal[0]]
        img_area_chenal_max = np.array(img_area_chenal_max, dtype='float32')
        out_chanel = img_area_chenal_max/255

    else:
        out_chanel = input_chanel
    return out_chanel


def seg_opt(output, mask=None):
    kernel = np.ones((3, 3), np.uint8)
    for i in range(0, 14):
        if mask is not None:
            gt = mask[i, :, :]
            gt = np.array(gt)
        chanel2 = output[i, :, :]

        if i != 0 and i != 1:
            if (i % 2) == 0:
                chanel2 = cv2.erode(chanel2, kernel, iterations=1)
                chanel2 = cv2.dilate(chanel2, kernel, iterations=1)
                dst = max_area_extract(chanel2)
            elif i == 13:
                dst = max_area_extract(chanel2)
            else:
                dst = max_area_extract(chanel2)
                rate = np.sum(dst)/np.sum(chanel2)
                if rate < 0.95:
                    chanel2 = cv2.erode(chanel2, kernel, iterations=1)
                    dst = cv2.dilate(chanel2, kernel, iterations=1)
            output[i, :, :] = dst
    return output