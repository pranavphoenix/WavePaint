import albumentations as A
from masks import get_mask_generator
import glob, os, cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import PIL.Image as Image

class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        # self.in_files = list(glob.glob(os.path.join(indir, '**', '*.JPEG'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)

def get_transforms(transform_variant, out_size):
    if transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')

    return transform

def make_default_train_dataloader(indir, kind='default', out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, **kwargs):

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)


    dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img
    


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')
    
class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.filenames = os.listdir(datadir)
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)
        # return len(self.filenames)

    def __getitem__(self, i):

        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

def make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs):
    if kind == 'default':
        dataset = InpaintingDataset(indir, **kwargs)

    return dataset

def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader

class PrecomputedInpaintingResultsDataset(InpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result
