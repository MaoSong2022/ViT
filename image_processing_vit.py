from typing import Optional, Dict, Union, List
import numpy as np
import PIL.Image

from torchvision.transforms import InterpolationMode


class ViTImageProcessor:

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[InterpolationMode] = InterpolationMode.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
    ):
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = (
            image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        )
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]

    def __call__(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        return self.preprocess(images, **kwargs)

    def resize(
        self, image: np.ndarray, size: Dict[str, int], resample: InterpolationMode
    ) -> np.ndarray:
        image = PIL.Image.fromarray(image)
        height, width = size["height"], size["width"]
        image = image.resize((width, height), resample=resample)
        return np.array(image)

    def rescale(
        self, image: np.ndarray, rescale_factor: Union[int, float]
    ) -> np.ndarray:
        input_type = image.dtype
        rescaled_image = image.astype(np.float64) * rescale_factor
        return rescaled_image.astype(input_type)

    def normalize(
        self,
        image: np.ndarray,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> np.ndarray:
        input_type = image.dtype
        image = image.astype(np.float32)
        # assume image is of shape [width, height, channels]
        num_channels = image.shape[2]
        if isinstance(image_mean, float):
            image_mean = [image_mean] * num_channels
        image_mean = np.array(image_mean, dtype=image.dtype)

        if isinstance(image_std, float):
            image_std = [image_std] * num_channels
        image_std = np.array(image_std, dtype=image.dtype)

        image = (image - image_mean) / image_std
        return image.astype(input_type)

    def preprocess(
        self,
        images: List[np.ndarray],
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[InterpolationMode] = InterpolationMode.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
    ) -> List[np.ndarray]:
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size

        # validate images are valid
        # validate arguments are valid

        # check if images are already been scaled

        # resize
        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, rescale_factor=rescale_factor)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, image_mean=image_mean, image_std=image_std)
                for image in images
            ]

        return images
