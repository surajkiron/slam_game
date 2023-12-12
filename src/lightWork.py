import torch
import sys
import numpy as np
import cv2
# from types import SimpleNamespace
from typing import List, Optional,  Union

sys.path.append("/home/harshit/vis_nav_player/finGame/src/LightGlue")
from lightglue import LightGlue, SuperPoint, DISK
# from lightglue.utils import load_image, rbd
# from lightglue import viz2d

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }

def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(image, resize: int = None, **kwargs) -> torch.Tensor:
    # image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

# def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
#     """Read an image from path as RGB or grayscale"""
#     if not Path(path).exists():
#         raise FileNotFoundError(f"No image at path {path}.")
#     mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
#     image = cv2.imread(str(path), mode)
#     if image is None:
#         raise IOError(f"Could not read image at {path}.")
#     if not grayscale:
#         image = image[..., ::-1]
#     return image

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def light(img0, img1):
    image0 = load_image(img0)
    image1 = load_image(img1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    # print(m_kpts0)
    # print(m_kpts1)
    m_kpts0 = m_kpts0.cpu().numpy()
    m_kpts1 = m_kpts1.cpu().numpy()
    return m_kpts0, m_kpts1

# img0 = read_image("/home/harshit/Downloads/vis_nav_player/images/1.jpg")
# img1 = read_image("/home/harshit/Downloads/vis_nav_player/images/2.jpg")

# # image0 = load_image(img0)
# # image1 = load_image(img1)

# # feats0 = extractor.extract(image0.to(device))
# # feats1 = extractor.extract(image1.to(device))
# # matches01 = matcher({"image0": feats0, "image1": feats1})
# # feats0, feats1, matches01 = [
# #     rbd(x) for x in [feats0, feats1, matches01]
# # ]  # remove batch dimension

# # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# m_kpts0, m_kpts1 = light(img0, img1)

# m_kpts0 = m_kpts0.cpu().numpy()
# m_kpts1 = m_kpts1.cpu().numpy()

# print(type(m_kpts0))
# print(type(m_kpts1))

