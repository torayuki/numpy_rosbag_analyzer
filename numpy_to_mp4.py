import os
import torch
import cv2
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ffmpeg


def tensor2numpy(array):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy().copy()
    return array


def flat(feat):
    feat_size = feat.shape[-1]
    return feat.reshape(-1, feat_size)


def get_xyz(feat):
    feat_flat = flat(feat)
    if not (feat_flat.shape[1] == 3):
        print("error")
    return feat_flat[:, 0], feat_flat[:, 1], feat_flat[:, 2]


def sound_postprocess(sound):
    sound_mel = np.multiply(sound, -80)
    return sound_mel


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth=5):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def plot_tragectory(data, ax, t, dim=3):
    pos_x, pos_y, pos_z = get_xyz(data[:t])
    if dim == 3:
        ax.plot(pos_x, pos_y, pos_z, marker="x")
    else:
        ax.plot(pos_x, pos_y, marker="x")
    ax.set_xlim(0.2, 0.3)
    ax.set_ylim(0.005, 0.01)
    if dim == 3:
        ax.set_zlim(0.15, 0.25)


def plot_observations(data, plot_dir, file_name, n_frame=None):
    if n_frame == None:
        n_frame = len(data["image_horizon"])

    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    fig = plt.figure(figsize=(4 * 2, 4 * 2))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")

    def plot(t):
        plt.cla()
        fig.suptitle("observation t={}".format(t))
        imh = postprocess_observation(data["image_horizon"][t])
        ax1.imshow(imh.transpose(1, 2, 0)[:, :, ::-1])
        mlsp = sound_postprocess(data["sound"][t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax2)
        plot_tragectory(data["end_effector"], ax3, t)
        ax1.title.set_text("Horizontal Image")
        ax2.title.set_text("Mel-Spectrogram")
        ax3.title.set_text("End-effector Position")

    # create animation 10Hz
    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_plot_name = os.path.join(plot_dir, file_name + ".mp4")
    anim.save(save_plot_name, writer="ffmpeg")


def main():
    numpy_path = "numpy/2022-03-24"
    numpy_dir_list = os.listdir(numpy_path)

    results_path = "result"
    os.makedirs(results_path, exist_ok=True)

    for numpy_dir in numpy_dir_list:
        print(numpy_dir)
        observations = {}
        dir_path = os.path.join(numpy_path, numpy_dir)
        n_frame = sum(os.path.isfile(os.path.join(dir_path, name)) for name in os.listdir(dir_path))
        for i in range(n_frame):
            file_path = os.path.join(dir_path, "frame_%05d.npy" % i)
            dict = np.load(file_path, allow_pickle=True).item()
            observation = dict["observations"]
            for key in observation.keys():
                if not key in observations.keys():
                    observations[key] = tensor2numpy(observation[key])
                else:
                    observations[key] = np.concatenate([observations[key], tensor2numpy(observation[key])], axis=0)
        # n_frame = 1
        plot_observations(observations, results_path, numpy_dir, n_frame)


if __name__ == "__main__":
    main()
