import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def penalty_gradient(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images, outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    normalized_gradient = gradient.norm(2, dim=1)
    penalty_gradient = torch.mean((normalized_gradient - 1) ** 2)
    return penalty_gradient


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for temp in optimizer.param_groups:
        temp["lr"] = lr


def plot_examples(low_resolution_folder, generator):
    files = os.listdir(low_resolution_folder)
    generator.eval()
    for file in files:
        image = Image.open("test_images/" + file)
        with torch.no_grad():
            upscale_img = generator(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscale_img * 0.5 + 0.5, f"saved/{file}")
    generator.train()
