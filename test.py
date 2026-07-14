# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: test.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import SimpleITK
import numpy as np
import os
import model
import utils
import torch
import argparse
from tqdm import tqdm
import data

if __name__ == '__main__':

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder', type=str, default='RDN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-pre_trained_model', type=str, default='./pre_trained_models/ArSSR_RDN.pkl',
                        dest='pre_trained_model', help='the file path of LR input image for testing')

    # about GPU
    parser.add_argument('-is_gpu', type=int, default=1, dest='is_gpu',
                        help='enable GPU (1->enable, 0->disenable)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')

    # about file
    parser.add_argument('-input_path', type=str, default='test/input', dest='input_path',
                        help='the file path of LR input image')
    parser.add_argument('-output_path', type=str, default='test/output', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('-scale', type=float, default='2.0', dest='scale',
                        help='the up-sampling scale k (isotropic, used when -sr_scale is not given)')
    parser.add_argument('-sr_axis', type=str, default='auto', dest='sr_axis',
                        help="anisotropic SR: which ARRAY axis (SimpleITK z,y,x order) to super-resolve. "
                             "'auto' = thickest axis (argmax of spacing), or 0/1/2. Used with -sr_scale.")
    parser.add_argument('-sr_scale', type=float, default=None, dest='sr_scale',
                        help="anisotropic SR: factor applied ONLY to -sr_axis (other axes x1). "
                             "Overrides isotropic -scale. e.g. -sr_axis auto -sr_scale 2.67 -> z-only 2.67x.")

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    pre_trained_model = args.pre_trained_model
    gpu = args.gpu
    is_gpu = args.is_gpu
    input_path = args.input_path
    output_path = args.output_path
    scale = args.scale
    sr_axis = args.sr_axis
    sr_scale = args.sr_scale

    # -----------------------
    # model
    # -----------------------
    if is_gpu == 1 and torch.cuda.is_available():
        DEVICE = torch.device('cuda:{}'.format(str(gpu)))
    else:
        DEVICE = torch.device('cpu')
    ArSSR = model.ArSSR(encoder_name=encoder_name,
                        feature_dim=feature_dim,
                        decoder_depth=int(decoder_depth / 2),
                        decoder_width=decoder_width).to(DEVICE)
    ArSSR.load_state_dict(torch.load(pre_trained_model, map_location=DEVICE))

    # -----------------------
    # SR
    # -----------------------
    filenames = os.listdir(input_path)
    for f in tqdm(filenames):
        # read the dimension size and spacing of LR input image
        lr = SimpleITK.ReadImage(r'{}/{}'.format(input_path, f))
        lr_size = SimpleITK.GetArrayFromImage(lr).shape          # array order (z,y,x)
        lr_spacing = lr.GetSpacing()                              # SimpleITK order (x,y,z)
        arr_spacing = np.array(lr_spacing)[::-1]                  # array order (z,y,x)

        # build per-axis scale vector (array order)
        if sr_scale is not None:
            ax = int(np.argmax(arr_spacing)) if sr_axis == 'auto' else int(sr_axis)
            scale_arr = np.ones(3, dtype=float)
            scale_arr[ax] = float(sr_scale)
            label = '{}x_axis{}'.format(sr_scale, ax)
        else:
            scale_arr = np.ones(3, dtype=float) * float(scale)
            label = str(scale)

        test_loader = data.loader_test(in_path_lr=r'{}/{}'.format(input_path, f), scale=scale_arr)
        # then compute the dimension size and spacing of the HR image
        hr_size = (np.array(lr_size) * scale_arr).astype(int)
        hr_spacing = np.array(lr_spacing) / scale_arr[::-1]       # back to SimpleITK (x,y,z) order

        ArSSR.eval()
        with torch.no_grad():
            img_pre = np.zeros((hr_size[0] * hr_size[1] * hr_size[2], 1))
            for i, (img_lr, xyz_hr) in enumerate(test_loader):
                img_lr = img_lr.unsqueeze(1).float().to(DEVICE)  # N×1×h×w×d
                xyz_hr = xyz_hr.view(1, -1, 3).float()  # N×K×3 (K=H×W×D)
                for j in tqdm(range(hr_size[0])):
                    xyz_hr_patch = xyz_hr[:, j * hr_size[1] * hr_size[2]:
                                             j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[2], :].to(DEVICE)
                    img_pre_path = ArSSR(img_lr, xyz_hr_patch)
                    img_pre[j * hr_size[1] * hr_size[2]:
                            j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[2]] = \
                        img_pre_path.cpu().detach().numpy().reshape(hr_size[1] * hr_size[2], 1)
                img_pre = img_pre.reshape((hr_size[0], hr_size[1], hr_size[2]))
        # save file
        utils.write_img(vol=img_pre,
                        ref_path=r'{}/{}'.format(input_path, f),
                        out_path=r'{}/ArSSR_{}_recon_{}_{}'.format(output_path, encoder_name,
                                                                    label.replace('.', 'd'), f),
                        new_spacing=hr_spacing)
