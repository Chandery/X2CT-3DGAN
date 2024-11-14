import hydra
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    LoadImage,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    EnsureType,
    Resized,
    SaveImage,
)
import monai.transforms as mt

from monai.networks.layers import Norm
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset


def prepare_dataset(data_path, resize_size, img_resize_size=None, cond_path=None, split="train"):
    """
    Prepare dataset for training
    data_path: str, path to nii data(3D)
    cond_path: str, path to x-ray 2d png images, if None means only conduct autoencoder process
    resize_size: tuple, (x, y, z)
    split: str, "train" or "val"
    """
    # if "data_path" in config.keys():
    data_list = sorted(Path(data_path).glob("*.nii*"))
    # print(data_list[0])
    if cond_path:
        cond_path = sorted(Path(cond_path).glob("*"))

    # * create data_dicts, a list of dictionary with keys "image" and "cond",  cond means x-ray 2d png image
    data_dicts = []
    if cond_path:
        for image, cond in zip(data_list, cond_path):
            tmp = {"image": image}
            cond_png = list(sorted(Path(cond).glob("*.jpg")))
            tmp["cond1"] = cond_png[0]
            tmp["cond2"] = cond_png[1]
            tmp["file_path"] = str(image)
            data_dicts.append(tmp)
    else:
        for image in data_list:
            tmp = {"image": image}
            data_dicts.append(tmp)

    # if split == "train":
    #     data_dicts = data_dicts[: int(len(data_dicts) * 0.8)]
    # else:
    #     data_dicts = data_dicts[int(len(data_dicts) * 0.8) :]

    if split == "train":
        data_dicts = data_dicts[: int(len(data_dicts) * 0.7)]
    elif split == "test":
        data_dicts = data_dicts[int(len(data_dicts) * 0.8) : ]
    elif split == "val":
        data_dicts = data_dicts[int(len(data_dicts) * 0.7) : int(len(data_dicts) * 0.8)]

    set_determinism(seed=0)

    if cond_path:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "cond1", "cond2"], ensure_channel_first=True),
                # Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                Resized(keys=["cond1", "cond2"], spatial_size=img_resize_size),
                # NormalizeIntensityd(keys=used_keys),
                # NormalizeIntensityd(keys=["image"]),
                ############## else ###############
                # ScaleIntensityd(keys=["imgae"]),
                ############## Fei  ###############
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
                ScaleIntensityd(keys=["cond1", "cond2"], minv=-1, maxv=1),
            ]
        )
    else:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
            ]
        )
    if split == "train":
        train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=0)
    else:
        train_ds = Dataset(data=data_dicts, transform=train_transforms)
    shuffle = False if split == "test" else True
    # train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, shuffle=shuffle)
    return train_ds, shuffle


# @hydra.main(config_path="../conf", config_name="config/TEST.yaml", version_base="1.3")
def main():
    # train_dl = prepare_dataset(
    #     data_path=config.data_path, resize_size=config.resize_size, cond_path=config.cond_path, split="test"
    # )
    test_ds, _shuffle= prepare_dataset(
        cond_path="/home/cdy/X2CT/X2CTData/result/split/penguDR"  ,
        data_path="/home/cdy/X2CT/X2CTData/pengu/all/",
        resize_size=[128,128,128],
        img_resize_size=[128,128],
        split="train",
    )

    from collate_fn import collate_gan_views

    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=_shuffle, collate_fn=collate_gan_views)

    for i in test_dl:
        # print(i.keys())
        # print(i["image"])
        # continue
        # print(i["cond1"].shape)
        img, cond, file_path = i
        cond1 = cond[0]
        # cond1 = i["cond1"]
        cond1 = cond1.permute(0, 2, 3, 1)
        cond1 = cond1 * 255
        cond = cond1[:, :, :, :3]
        cond = cond * 255
        print(cond.shape)
        # img = i["image"]
        # img = img.squeeze(0)
        # img = img * 255
        img = (img + 1) * 127.5
        print(img.shape)
        saver_origin = SaveImage(
            output_dir="./output_nii",
            output_ext=".nii.gz",
            # output_ext=".nii",
            output_postfix="cache",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_origin(img)

        saver = SaveImage(
            output_dir="./",
            output_ext=".png",
            output_postfix="PIL",
            output_dtype=np.uint8,
            resample=False,
            squeeze_end_dims=True,
            writer="PILWriter",
        )
        img = saver(cond1)
        break


def test_save_image():
    # img = LoadImage()
    path = "/disk/ssy/data/drr/feijiejie/all/LNDb-0210.nii"
    trans = Compose(
        [
            LoadImaged(keys="image", ensure_channel_first=True),
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            Resized(keys="image", spatial_size=(128, 128, 128)),
            # ScaleIntensityd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
        ]
    )
    d = {"image": path}
    img = trans(d)
    img = img["image"]
    # print(img.shape)
    # print(img.affine)
    # print(img.meta.keys())
    img = (img + 1) * 127.5
    # print(img.shape)
    saver_origin = SaveImage(
        output_dir="./",
        output_ext=".nii.gz",
        output_postfix="origin",
        separate_folder=False,
        output_dtype=np.uint8,
        # scale=255,
        resample=False,
        squeeze_end_dims=True,
        writer="NibabelWriter",
    )
    saver_origin(img)


if __name__ == "__main__":
    main()
    # test_save_image()
