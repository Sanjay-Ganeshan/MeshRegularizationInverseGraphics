import torch
import torch.nn as nn
import pyredner as pyr
from ..util.imageops import GaussianBlur, HalfSize

class ImagePyramidLoss(nn.Module):
    def __init__(self, nlevels = 3):
        super().__init__()
        # For calculating loss per level
        self.mse = nn.MSELoss()
        self.weights = [1] * nlevels
        self.nlevels = nlevels
        self.blur = GaussianBlur(20, 21).to(pyr.get_device())
        self.halfer = HalfSize()

    def forward(self, original_image, target_image):
        orig = original_image
        target = target_image
        total_loss = self.mse(orig, target)
        for each_level in range(self.nlevels):
            # Blur
            orig = self.blur(orig)
            target = self.blur(target)
            # Scale
            orig = self.halfer(orig)
            target = self.halfer(target)
            # Add current loss
            total_loss += self.mse(orig, target)
        return total_loss

