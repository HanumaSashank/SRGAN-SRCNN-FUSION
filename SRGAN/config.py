import torch
from PIL import Image
import albumentations
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARN_RATE = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 80
LOW_RES = int(HIGH_RES / 4)
IMG_CHANNELS = 3

low_resolution_transform = albumentations.Compose(
    [
        albumentations.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

high_resolution_transform = albumentations.Compose(
    [
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

both_resolution_transforms = albumentations.Compose(
    [
        albumentations.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),
    ]
)

test_transform = albumentations.Compose(
    [
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
