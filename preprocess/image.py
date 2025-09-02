"""Image preprocessing utilities."""

from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional

# Mean and std for normalization using ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_image_transform(size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Return a torchvision transform for resizing and normalizing images."""
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: str, transform: Optional[transforms.Compose] = None) -> transforms.Tensor:
    """Load an image from ``path`` and apply ``transform``.

    Parameters
    ----------
    path: str
        File path to the image.
    transform: torchvision.transforms.Compose, optional
        Transformation pipeline to apply. If ``None``, :func:`default_image_transform`
        is used.
    """
    img = Image.open(path).convert("RGB")
    transform = transform or default_image_transform()
    return transform(img)

__all__ = ["load_image", "default_image_transform", "IMAGENET_MEAN", "IMAGENET_STD"]
