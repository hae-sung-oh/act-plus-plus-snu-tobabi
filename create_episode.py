import argparse
import os
import time
import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import Optional, Union, List
import h5py
from torch import Tensor
import pickle


def convert_data(*data):
    output = []
    for item in data:
        if isinstance(item, list):
            output.append(np.array(item))
        elif isinstance(item, ndarray):
            output.append(item)
        elif isinstance(item, Tensor):
            output.append(item.cpu().numpy())
        elif isinstance(item, str):
            if not item.endswith((".pkl", ".npy")):
                raise ValueError("Please provide .pkl or .npy file")
            if not os.path.isfile(item):
                raise FileNotFoundError("File not found")
            if item.endswith(".pkl"):
                with open(item, "rb") as f:
                    temp = pickle.load(f)
                output.append(convert_data(temp)[0])
            else:
                output.append(np.load(item))
        else:
            raise TypeError("Unsupported type")
    return output


def extract_data(image_root_dir):
    images = []
    camera_names = []

    try:
        with os.scandir(image_root_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    camera_names.append(entry.name)
                    camera_images = []
                    image_folder = os.path.join(image_root_dir, entry.name)
                    with os.scandir(image_folder) as img_entries:
                        sorted = sorted(img_entries, key=lambda e: int(os.path.splitext(e.name)[0]))
                        for img_entry in sorted:
                            if img_entry.is_file() and img_entry.name.endswith((".jpg", ".jpeg", ".png")):
                                image_path = os.path.join(image_folder, img_entry.name)
                                image = cv2.imread(image_path)
                                if image is not None:
                                    camera_images.append(image)
                    images.append(np.array(camera_images))
    except FileNotFoundError:
        print(f"The directory {image_root_dir} does not exist.")
    except PermissionError:
        print(f"Permission denied to access {image_root_dir}.")

    return images, camera_names


def create_episode(
    images: Optional[Union[List[ndarray], ndarray[int], Tensor[int], str]],
    qpos: Union[List[float], ndarray[float], Tensor[float], str],
    qvel: Union[List[float], ndarray[float], Tensor[float], str],
    effort: Union[List[float], ndarray[float], Tensor[float], str],
    action: Union[List[float], ndarray[float], Tensor[float], str],
    base_action: Union[List[float], ndarray[float], Tensor[float], str],
    image_root_dir: Optional[str],
    camera_names: Optional[List[str]],
    dataset_dir: str,
    dataset_name: str,
    overwrite: bool = False,
    no_compress: bool = False,
):
    """
    Creates a Mobile-Aloha dataset for n timesteps, containing observations and actions.

    Observations:
        - images (from m cameras) `m x n x (480, 640, 3)`
            - Each camera directory contains images of shape `n x (480, 640, 3)` with dtype <int8>.
            - Example camera directory structure:
                - cam_high: `n x (480, 640, 3)` <int8>
                - ... (other cameras)

        - qpos: `n x (14,)` <float>
        - qvel: `n x (14,)` <float>
        - effort:  `n x (14,)` <float>

    Actions:
        - action: `n x (14,)`  <float>
        - base_action: `n x (2,)`  <float>

    Parameters:
        - images (List[ndarray], ndarray[int], Tensor[int], str, or None):
            - Image data.
        - qpos (List[float], ndarray[float], Tensor[float], or str):
            - Position data.
        - qvel (List[float], ndarray[float], Tensor[float], or str):
            - Velocity data.
        - effort (List[float], ndarray[float], Tensor[float], or str):
            - Effort data.
        - action (List[float], ndarray[float], Tensor[float], or str):
            - Action data.
        - base_action (List[float], ndarray[float], Tensor[float], or str):
            - Base action data.
        - image_root_dir (str or None):
            - Root directory for images (overwrites `images` argument if provided).
        - camera_names (List[str] or None):
            - List of camera names. Length should be equal to the number of cameras (m).
        - dataset_dir (str):
            - Directory where the dataset will be stored.
        - dataset_name (str):
            - Name of the dataset.
        - overwrite (bool):
            - Whether to overwrite existing dataset. Default is False.
        - no_compress (bool):
            - Whether to compress the dataset. Default is False.

    Output:
        - A .hdf5 file for an episode.

    Directory structure for `image_root_dir` (if provided):
        - image_root_dir
            - camera_name_0
                - 0.jpg
                - 1.jpg
                - 2.jpg
                - ...
                - n.jpg
            - camera_name_1
                - 0.jpg
                - 1.jpg
                - 2.jpg
                - ...
                - n.jpg
            - ...
    """
    if image_root_dir is None and images is None:
        raise ValueError("You have to provide whether 'images' or 'image_root_dir' argument")
    if image_root_dir is not None:  # Import images from the directory
        images, camera_names = extract_data(image_root_dir)

    images, qpos, qvel, effort, action, base_action = convert_data(images, qpos, qvel, effort, action, base_action)

    input_data_shapes = [images.shape[1]] + [item.shape[0] for item in [qpos, qvel, effort, action, base_action]]

    assert len(set(input_data_shapes)) == 1, f"Input data shapes are not matched: {input_data_shapes}"
    assert images.shape[0] == len(camera_names), f"Number of camera not matched: images-{images.shape[0]}, camera-{len(camera_names)}"

    max_timesteps = input_data_shapes[0]

    print(f"Dataset name: {dataset_name}")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.")
        exit()

    data_dict = {
        "/observations/qpos": qpos,
        "/observations/qvel": qvel,
        "/observations/effort": effort,
        "/action": action,
        "/base_action": base_action,
    }
    for i, cam_name in enumerate(camera_names):
        data_dict[f"/observations/images/{cam_name}"] = images[i]

    if not no_compress:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode(".jpg", image, encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        print(f"compression: {time.time() - t0:.2f}s")

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f"/observations/images/{cam_name}"]
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f"/observations/images/{cam_name}"] = padded_compressed_image_list
        print(f"padding: {time.time() - t0:.2f}s")

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = not no_compress
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if not no_compress:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, padded_size),
                    dtype="uint8",
                    chunks=(1, padded_size),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 14))
        _ = root.create_dataset("base_action", (max_timesteps, 2))
        # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

        for name, array in data_dict.items():
            root[name][...] = array

        if not no_compress:
            _ = root.create_dataset("compress_len", (len(camera_names), max_timesteps))
            root["/compress_len"][...] = compressed_len

    print(f"Saving: {time.time() - t0:.1f} secs")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Image data or path to image data directory.")
    parser.add_argument("--qpos", type=str, required=True, help="Path to position data file.")
    parser.add_argument("--qvel", type=str, required=True, help="Path to velocity data file.")
    parser.add_argument("--effort", type=str, required=True, help="Path to effort data file.")
    parser.add_argument("--action", type=str, required=True, help="Path to action data file.")
    parser.add_argument("--base_action", type=str, required=True, help="Path to base action data file.")
    parser.add_argument("--image_root_dir", type=str, help="Root directory for images (overwrites images argument if provided).")
    parser.add_argument("--camera_names", nargs="+", help="List of camera names.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset will be stored.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--overwrite", action="store_false", help="Whether to overwrite existing dataset.")
    parser.add_argument("--no_compress", action="store_false", help="Whether to compress the dataset.")

    args = parser.parse_args()

    create_episode(
        images=args.images,
        qpos=args.qpos,
        qvel=args.qvel,
        effort=args.effort,
        action=args.action,
        base_action=args.base_action,
        image_root_dir=args.image_root_dir,
        camera_names=args.camera_names,
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        no_compress=args.no_compress,
    )


if __name__ == "__main__":
    main()
