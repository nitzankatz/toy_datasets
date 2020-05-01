import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


def make_ellipse(im_len, ellipse_x, ellipse_y, rot=0, gray_level=1, blur=None):
    im = np.zeros((im_len, im_len), dtype=float)
    alpha = np.deg2rad(rot)
    cos = np.cos(alpha)
    sin = np.sin(alpha)
    center = im_len // 2
    lin = np.array(range(im_len))
    xx, yy = np.meshgrid(lin, lin)
    loc1 = (xx - center) * cos - (yy - center) * sin
    loc2 = (xx - center) * sin + (yy - center) * cos
    im[(loc1) ** 2 / ellipse_x ** 2 + (loc2) ** 2 / ellipse_y ** 2 < 1] = 1
    if blur is not None:
        im = gaussian_filter(im, sigma=blur)
    return im


def create_dataset(dir, im_len, x_range, y_range, rot_range=[0], gray_range=[1], blur=None):
    vis_dir = dir + '_vis'
    os.makedirs(dir, exist_ok=True)
    os.makedirs(vis_dir,exist_ok=True)
    for x in x_range:
        for y in y_range:
            for rot in rot_range:
                for gray in gray_range:
                    im_name = '{}_{}_{}_{}'.format(x, y, rot, gray)
                    im_path = os.path.join(dir, im_name + '.npy')
                    im_path_show = os.path.join(vis_dir, im_name + '.png')
                    im = make_ellipse(im_len, x, y, rot, gray, blur)
                    np.save(im_path, im)
                    plt.imsave(im_path_show, im, cmap='gray', vmin=0, vmax=1)


if __name__ == '__main__':
    # im = make_ellipse(54, 10, 27, rot=37, gray_level=0.5)
    # plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.show()
    create_dataset('data1', 20, range(10), range(10),rot_range=[0, 20])
