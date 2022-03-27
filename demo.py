import os
import cv2
import time
import numpy as np
from matting import Matting

if __name__ == '__main__':
    # Image directory.
    img_dir         = './img'
    # Result directory.
    out_comp_dir    = './out/comp'
    out_matte_dir   = './out/matte'
    out_rgba_dir    = './out/rgba'

    bg_color = [33, 150, 243]  # Background color (BGR format).

    os.makedirs(out_comp_dir, exist_ok=True)
    os.makedirs(out_matte_dir, exist_ok=True)
    os.makedirs(out_rgba_dir, exist_ok=True)

    M = Matting(gpu=True)  # initial model.

    img_names = os.listdir(img_dir)

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)

        # matting.
        matte, rgba = M.matting(img_path, max_size=378)

        # composition.
        comp = M.composite(rgba, np.array(bg_color) / 255.)

        # save.
        img_name = img_name + '.png'  # for RGBA image.
        cv2.imwrite(os.path.join(out_matte_dir, img_name), np.uint8(matte * 255))
        cv2.imwrite(os.path.join(out_rgba_dir, img_name), cv2.cvtColor(np.uint8(rgba * 255), cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(os.path.join(out_comp_dir, img_name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))
