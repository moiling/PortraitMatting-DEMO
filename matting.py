import math

import torch
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F


class Matting:
    def __init__(self, model_path='./ckpt/model.pt', gpu=True):
        torch.set_flush_denormal(True)  # flush cpu subnormal float.
        self.model_path = model_path
        self.gpu = gpu
        self.model = self.__load_model()

    def __load_model(self):
        model = torch.jit.load(self.model_path, map_location='cpu')
        if self.gpu and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        model.eval()
        return model

    def matting(self, image_path, max_size=-1):
        """
        :param   image_path : Image file path.
        :param   max_size   : Maximum size of output image. (max_size <= 0 means that the original size is not changed.)
        :return: 
                 pred_matte : shape: [H, w, 1      ] range: [0, 1]
                 rgba       : shape: [H, W, RGBA(4)] range: [0, 1]
        """
        with torch.no_grad():
            image = self.__load_image_tensor(image_path, max_size)
            if self.gpu and torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.cpu()

            b, c, h, w = image.shape

            # resize to training size.
            resize_image = F.resize(image, [320, 320], Image.BILINEAR)
            pred_matte = self.model(resize_image)
            pred_matte = F.resize(pred_matte, [h, w])
            pred_matte = pred_matte.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            image = image.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            pred_rgba = self.cutout(image, pred_matte)
            
            return pred_matte, pred_rgba


    @staticmethod
    def cutout(image, alpha):
        """
        :param   image: shape: [H, W, RGB(3) ] range: [0, 1]
        :param   alpha: shape: [H, w, 1      ] range: [0, 1]
        :return       : shape: [H, W, RGBA(4)] range: [0, 1]
        """
        cutout = np.zeros((image.shape[0], image.shape[1], 4))
        cutout[..., :3] = image
        cutout[...,  3] = alpha.astype(np.float32).squeeze(axis=2)       # [H, W, RGBA(4)]
        return cutout

    @staticmethod
    def composite(cutout, bg):
        """
        :param  cutout: shape: [H, W, RGBA(4)] range: [0, 1]
        :param  bg    : shape: [BGR(3)]        range: [0, 1]
        :return       : shape: [H, W, RGB(3) ] range: [0, 1]
        """
        alpha = cutout[:, :, 3:4]
        fg    = cutout[:, :,  :3]
        image = alpha * fg + (1 - alpha) * bg
        return image

    def __load_image_tensor(self, image_path, max_size=-1):
        image = Image.open(image_path).convert('RGB')
        if max_size > 0:
            [image] = ResizeIfBiggerThan(max_size)([image])
        [image] = ToTensor()([image])
        image = image.unsqueeze(dim=0)
        return image
    

class ResizeIfBiggerThan(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        for idx, image in enumerate(images):
            max_size = max(image.size)
            if max_size > self.size:
                rate = self.size / float(max_size)
                h, w = math.ceil(rate * image.size[0]), math.ceil(rate * image.size[1])
                images[idx] = F.resize(image, [w, h])
        return images


class ToTensor(object):
    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.to_tensor(image)
        return images
