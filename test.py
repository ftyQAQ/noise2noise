import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
import os
import dxchange
from skimage import measure,metrics

def nomal0(image0):
    image0 = image0.astype(np.float32)
    x_mean = np.mean(image0)
    x_std = np.std(image0)
    # image0 = (image0 - x_mean) / np.maximum(x_std, 1e-7)
    x_max = np.max(image0)
    x_min = np.min(image0)
    image = (image0-x_min)/(x_max-x_min)*255.
    return image
def red_stack_tiff_nomal(path):
    files = os.listdir(path)
    prj = []
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            p = nomal0(p)
            prj.append(p)
    pr = np.array(prj)
    return pr
def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args(['--image_dir=G:/noise2noise/data_tiff/test/',
                              '--weight_file=G:/noise2noise/cell1/weights.001-37.043-30.04738.hdf5',
                              '--output_dir=weight_file=G:/noise2noise/cell1/test_out1/'])
    return args
def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)
def nomal(image0):

    image0 = image0.astype(np.float32)
    x_mean = np.mean(image0)
    x_std = np.std(image0)
    image0 = (image0 - x_mean) / np.maximum(x_std, 1e-7)
    x_max = np.max(image0)
    x_min = np.min(image0)
    image0 = (image0-x_min)/(x_max-x_min)*255.
    h, w= image0.shape
    image = np.zeros((h, w,3))
    image[:,:,0] = image0
    image[:, :, 1] = image0
    image[:, :, 2] = image0

    return image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff",".tif",".hdf5"])
def read_tiff3(path):
    files = os.listdir(path)
    prj = []
    # prj0 = np.zeros((len(files), size, size))
    for n, file in enumerate(files):
        if is_image_file(file):
            p0 = dxchange.read_tiff(path + file)
            p = nomal(p0)
            prj.append(p)
    pr = np.array(prj)
    return pr
def main():
    # args = get_args()
    image_dir = r'G:\noise2noise\CT\test/'
    weight_file = r'G:\noise2noise\CT\model_u_net_fft2/'
    output_dir0 = r'G:\noise2noise\CT\model_u_net_fft2_out/'
    x_norm = read_tiff2(image_dir)
    files = os.listdir(weight_file)
    for i, file in enumerate(files):
        if is_image_file(file):
            model = get_model("unet")
            model.load_weights(weight_file+file)
            y_pred = model.predict(x_norm, batch_size=8, verbose=0)
            fil = 'cell_'
            output_dir = output_dir0+'%s/'%file
            # ph_re = r'G:\NMC\nmc2\all_input_zx_crack94_lab0/'
            y_pred = y_pred.astype(np.float32)
            for i_na, re in enumerate(y_pred):
                if i_na < 10:
                    dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s000%s_0.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s000%s_1.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s000%s_2.tiff' % (fil, i_na))
                elif 9 < i_na < 100:

                    dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s00%s_0.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s00%s_1.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s00%s_2.tiff' % (fil, i_na))
                elif 99 < i_na < 1000:
                    dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s0%s_0.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s0%s_1.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s0%s_2.tiff' % (fil, i_na))
                else:
                    dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s%s_0.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s%s_1.tiff' % (fil, i_na))
                    dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s%s_2.tiff' % (fil, i_na))

    # val_noise_model = get_noise_model(args.test_noise_model)
def nomal2(image0):

    image0 = image0.astype(np.float32)
    # x_mean = np.mean(image0)
    # x_std = np.std(image0)
    # image0 = (image0 - x_mean) / np.maximum(x_std, 1e-7)
    x_max = np.max(image0)
    x_min = np.min(image0)
    image0 = image0/(x_max-x_min)*255.
    h, w,_= image0.shape
    image = np.zeros((h, w,3))
    # image[:,:,0] = image0
    # image[:, :, 1] = image0
    # image[:, :, 2] = image0
    image[:, :, :2] = image0


    return image

def read_tiff2(path):
    files = os.listdir(path)
    prj = []
    # prj0 = np.zeros((len(files), size, size))
    for n, file in enumerate(files):
        if is_image_file(file):
            p0 = dxchange.read_tiff(path + file)
            p = nomal2(p0)
            prj.append(p)
    pr = np.array(prj)
    return pr
def main_single():
    # args = get_args()
    image_dir = r'G:\noise2noise\nmc_fbp_90_rot\test/'
    weight_file = r'G:\noise2noise\nmc_fbp_90_rot\model_u_net/u_net_005.hdf5'
    output_dir = r'G:\noise2noise\nmc_fbp_90_rot\model_u_net_out_test/'
    x_norm = read_tiff2(image_dir)
    model = get_model("unet")
    model.load_weights(weight_file)
    y_pred = model.predict(x_norm, batch_size=8, verbose=0)
    fil = 'cell_'
    # output_dir = output_dir0 + '%s/' % file
    # ph_re = r'G:\NMC\nmc2\all_input_zx_crack94_lab0/'
    y_pred = y_pred.astype(np.float32)
    for i_na, re in enumerate(y_pred):
        if i_na < 10:
            dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s000%s_0.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s000%s_1.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s000%s_2.tiff' % (fil, i_na))
        elif 9 < i_na < 100:

            dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s00%s_0.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s00%s_1.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s00%s_2.tiff' % (fil, i_na))
        elif 99 < i_na < 1000:
            dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s0%s_0.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s0%s_1.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s0%s_2.tiff' % (fil, i_na))
        else:
            dxchange.writer.write_tiff(re[:, :, 0], output_dir + '%s%s_0.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 1], output_dir + '%s%s_1.tiff' % (fil, i_na))
            dxchange.writer.write_tiff(re[:, :, 2], output_dir + '%s%s_2.tiff' % (fil, i_na))


    # val_noise_model = get_noise_model(args.test_noise_model)
def red_stack_tiff(path):
    # path0 = 'D:/pycharm/pycharm/py/resig/data/shapp3d_160/'
    files = os.listdir(path)
    prj = []
    # prj0 = np.zeros((len(files), size, size))
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            prj.append(p)
    pr = np.array(prj)
    return pr
def make_eavl_orgin():
    # path1 = r'G:\noise_data\cell\or_nomal/'
    # path2 = r'G:\noise2noise\cellpro\out\u_net_024.hdf5/'
    # ph_re = r'G:\noise2noise\cellpro/bg/'
    # p1 = dxchange.read_tiff(path1 + 'noise_input1_0024.tiff')
    # p2 = dxchange.read_tiff(path2 + 'cell_0024_1.tiff')
    # p0 = p2-p1
    # dxchange.write_tiff(p0,ph_re + 'bg_0024.tiff')
    from sklearn.metrics import mean_squared_error
    path1 = r'G:\noise_data\nmc\tomo_moments_fbp/'
    path2 = r'G:\noise_data\nmc\all_90_step_2\tomo2_fbp1/'
    # ph_re = r'G:\noise2noise\cellpro/bg/'
    # p1 = dxchange.read_tiff(path1 + 'noise_input1_0024.tiff')
    # p2 = dxchange.read_tiff(path2 + 'bg_0024.tiff')
    # dark = np.zeros((512,512))
    # p0 = tomopy.normalize(p1, p2, dark)
    # dxchange.write_tiff(p0, ph_re + 'divi_0024.tiff')
    grouds = red_stack_tiff_nomal(path1)
    # grouds = nomal(grouds)
    psnrs = []
    ssims = []
    mses = []
    mses1 = []
    predicts = red_stack_tiff_nomal(path2)
    for i, (groud, predict) in enumerate(zip(grouds, predicts)):
        # groud = np.clip(groud, 0, 255).astype(np.uint8)
        # predict = np.clip(predict, 0, 255).astype(np.uint8)
        psnr = metrics.peak_signal_noise_ratio(groud, predict, data_range=255)
        ssim = metrics.structural_similarity(groud, predict, data_range=255)
        mse = metrics.mean_squared_error(groud, predict)
        mse1 = mean_squared_error(groud, predict)
        ssims.append(ssim)
        psnrs.append(psnr)
        mses.append(mse)
        mses1.append(mse1 )
        print('i:',i,'psnr:', psnr, 'ssim:', ssim,'mse',mse,'mse1',mse1)
    print(np.mean(psnrs), np.mean(ssims), np.mean(mses), np.mean(mses1))
    #31.929937110478033 0.8014560450809928 55.829027752977005
    #31.9299391247435 0.8014560477582949 55.82902022405821
if __name__ == '__main__':
    main()