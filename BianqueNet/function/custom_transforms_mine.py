import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize_img(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
        @ 具体数值来自train文件夹
    """
    def __init__(self, mean=(0.2703814, 0.2703814, 0.2703814), std=(0.24898738, 0.24898738, 0.24898738)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((1, 2, 0))
        # mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class ToTensor_img(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        # mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        # mask = torch.from_numpy(mask).float()

        return img


class ToTensor_mask(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(mask).astype(np.float32).transpose((2, 0, 1))
        # mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        # mask = torch.from_numpy(mask).float()

        return img


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.17801675, 0.17801675, 0.17801675), std=(0.17522617, 0.17522617, 0.17522617)):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = np.array(img).astype(np.float32).transpose((1, 2, 0))
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        return img, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return img, mask


class Normalize_Totensor(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.18074335, 0.18074335, 0.18074335), std=(0.18184102, 0.18184102, 0.18184102)):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = np.array(img).astype(np.float32)
        # mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        return img, mask


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, mask


class RandomGaussianBlur_img(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, mask):
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.95), int(self.base_size * 1.05))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img, mask):

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return img, mask


