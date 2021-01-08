import torch
import torch.nn as nn

class ImageSilhoutteLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_image, target_image):
        # These are BHWC
        # Extract just the alpha channel
        orig_alpha = original_image[:,:,:,3]
        target_alpha = target_image[:,:,:,3]

        # Stack them so we can use max and min
        stacked = torch.stack([orig_alpha, target_alpha], dim=0)

        # Max = 1 if it's in either
        in_either_image, _ = torch.max(stacked, dim=0)
        # Min = 1 if it's in both
        in_both_images, _ = torch.min(stacked, dim=0)

        # Calculate in pix
        n_pix_in_intersection = torch.sum(in_both_images)
        n_pix_in_union = torch.sum(in_either_image)
        
        # Find the ratio. This'll be somewhere between 0 and 1
        # We dirty-divide to avoid div by 0
        iou = n_pix_in_intersection / (n_pix_in_union + 0.0000001)

        # Ideally intersection = union .. iou = 1
        # So flip

        return (iou * -1) + 1
        

