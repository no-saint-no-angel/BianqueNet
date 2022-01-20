import cv2
import time
import os
import copy
import network_big_apex.network_deeplab_upernet_aspp_psp_ab_swin_skips_1288 as network
import pandas as pd
from function.custom_transforms_mine import *
from function.segmentation_optimization import seg_opt
from function.calcu_DHI_512 import calcu_DHI
from function.calcu_signal_strength import calcu_Sigs
from function.quantitative_analysis import quantitative_analysis
from function.shiyan_jihe_signal_mean_std_plot_function import scatter_mean_std
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_start = time.time()


class DualCompose_img:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


image_only_transform = DualCompose_img([
    ToTensor_img(),
    Normalize_img()
])


def clahe_cv(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output_cv = cv2.merge([b, g, r])
    return output_cv

model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

model = model_map['deeplabv3plus_resnet101'](num_classes=14, output_stride=16)
model = torch.nn.DataParallel(model)
# load model weights
model_weight_path = "./weights_big_apex/deeplab_upernet_aspp_psp_ab_swin_skips_1288/deeplab_upe" \
                    "rnet_aspp_psp_ab_swin_skips_1288_0.0003.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
model.eval()
data_input_path = "./input/data_input"
results_output_path = "./output"
quantitative_analysis_results_output_name = 'quantitative_analysis_results' + '.xlsx'

dirList = os.listdir(data_input_path)

with torch.no_grad():
    for dir in dirList:
        data_input_dir = os.path.join(data_input_path, dir)
        data_output_path = os.path.join(results_output_path, dir)
        img_list = os.listdir(data_input_dir)
        for im_name in img_list:
            im_name_no_suffix = (im_name.split('.')[0]).split('-')[-1]
            input_age = int(im_name_no_suffix[0:2])
            input_sex = int(im_name_no_suffix[3])
            im_path = os.path.join(data_input_dir, im_name)
            print('processing ' + str(im_path) + '.'*20)
            input = cv2.imread(im_path)
            out_cv = clahe_cv(input)
            input_img = image_only_transform(out_cv)
            input_img = torch.unsqueeze(input_img, 0)
            pred_img = model(input_img)
            output = torch.nn.Softmax2d()(pred_img)
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            output_seg_opt = output.clone()
            output_seg_opt = torch.squeeze(output_seg_opt).numpy()
            output = seg_opt(output_seg_opt)
            try:
                jihe_parameter = []
                time_calcu_DHI_bf = time.time()
                DHI, DWR, HD, HV, point_big_h, point_big_w, point_fenge_h_big, point_fenge_w_big = calcu_DHI(output)
                jihe_parameter.append(HD)
                jihe_parameter.append(DHI)
                jihe_parameter.append(DWR)

                time_calcu_DHI_af = time.time()
                time_calcu_DHI = time_calcu_DHI_af - time_calcu_DHI_bf

                point_big_h = np.array(point_big_h)
                point_big_h = point_big_h.flatten()
                point_big_w = np.array(point_big_w)
                point_big_w = point_big_w.flatten()
                point_input_pic = copy.deepcopy(input)
                point_size = 1
                point_color = (0, 0, 255)  # BGR
                thickness = 4

                for p in range(len(point_big_h)):
                    point = (point_big_w[p], point_big_h[p])
                    cv2.circle(point_input_pic, point, point_size, point_color, thickness)

                point_fenge_h_big = np.array(point_fenge_h_big)
                point_fenge_h_big = point_fenge_h_big.flatten()
                point_fenge_w_big = np.array(point_fenge_w_big)
                point_fenge_w_big = point_fenge_w_big.flatten()
                point_size = 1
                point_color = (0, 255, 0)  # BGR
                thickness = 4

                for s in range(len(point_fenge_w_big)):
                    point = (point_fenge_w_big[s], point_fenge_h_big[s])
                    cv2.circle(point_input_pic, point, point_size, point_color, thickness)
                cv2.imshow('point_input_pic', point_input_pic)
                cv2.imwrite(os.path.join(data_output_path, "point_detect.BMP"), point_input_pic)

                SI_parameter = []
                time_calcu_Sigs_bf = time.time()
                inputs_Sigs = input
                output_Sigs = output.copy()
                SI_big_final, disc_si_dif_final = calcu_Sigs(inputs_Sigs, output_Sigs)
                SI_parameter.append(disc_si_dif_final)

                time_calcu_Sigs_af = time.time()
                time_calcu_Sigs = time_calcu_Sigs_af - time_calcu_Sigs_bf

                scatter_mean_std(data_output_path, input_sex, input_age, disc_si_dif_final, HD, DHI, DWR)

                quantitative_results = quantitative_analysis(disc_si_dif_final, HD, DHI, DWR, input_sex)

                data_jihe_parameter = pd.DataFrame(jihe_parameter)
                data_SI_parameter = pd.DataFrame(SI_parameter)
                data_quantitative_results = pd.DataFrame(quantitative_results)

                quantitative_analysis_results_output_name_path = os.path.join(data_output_path, str(
                    im_name.split('.')[0]) + quantitative_analysis_results_output_name)
                writer = pd.ExcelWriter(quantitative_analysis_results_output_name_path)
                data_jihe_parameter.to_excel(writer, 'jihe_parameter', float_format='%.5f')
                data_SI_parameter.to_excel(writer, 'SI_parameter', float_format='%.5f')
                data_quantitative_results.to_excel(writer, 'quantitative_results', float_format='%.5f')
                writer.save()
                writer.close()

            except Exception as e:
                print("---------------------------------------------------------the calculation of " + str(im_path) + " picture is failed!")
                pass
            continue

