import os
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


def convine_mp4_wav(video_path, wav_path, out_path, overwite=True):
    instream_v = ffmpeg.input(video_path)
    instream_a = ffmpeg.input(wav_path)
    stream = ffmpeg.output(instream_v, instream_a, out_path, vcodec="copy", acodec="aac")
    ffmpeg.run(stream, overwrite_output=overwite, quiet=True)


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

    fig = plt.figure(figsize=(3 * 2, 3 * 2))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    def plot(t):
        plt.cla()
        fig.suptitle("observation t={}".format(t))
        ax1.imshow(data["image_horizon"][t])
        ax2.imshow(data["image_vertical"][t])
        mlsp = sound_postprocess(data["sound"][t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax3)
        plot_tragectory(data["end_effector"], ax4, t)
        ax1.title.set_text("Horizontal Image")
        ax2.title.set_text("Vertical Image")
        ax3.title.set_text("Mel-Spectrogram")
        ax4.title.set_text("End-effector Position")

    # create animation 10Hz
    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_plot_name = os.path.join(plot_dir, file_name + ".mp4")
    anim.save(save_plot_name, writer="ffmpeg")


def main():
    train_dataset_dir = "../dataset/PointDrilling20220111/train_dataset"
    test_dataset_dir = "../dataset/PointDrilling20220111/test_dataset"
    train_bag_dir = os.path.join(train_dataset_dir, "bag")
    train_pack_dir = os.path.join(train_dataset_dir, "pack")
    train_plot_dir = os.path.join(train_dataset_dir, "plot")
    test_bag_dir = os.path.join(test_dataset_dir, "bag")
    test_plot_dir = os.path.join(test_dataset_dir, "plot")
    test_pack_dir = os.path.join(test_dataset_dir, "pack")

    os.makedirs(train_plot_dir, exist_ok=True)
    os.makedirs(test_plot_dir, exist_ok=True)

    # find train npy files
    train_npy_list = glob.glob(os.path.join(train_pack_dir, "**.npy"))
    print("find %d train bags!" % len(train_npy_list))
    # train process
    for npy_path in tqdm(train_npy_list, desc="train dataset"):
        file_name = os.path.splitext(os.path.basename(npy_path))[0]
        npy_dict = np.load(npy_path, allow_pickle=True).item()
        plot_observations(npy_dict, train_plot_dir, file_name)
        # TODO: the length of mp4 and wav is different
        convine_mp4_wav(
            os.path.join(train_plot_dir, file_name + ".mp4"),
            os.path.join(train_bag_dir, file_name, file_name + ".wav"),
            os.path.join(train_plot_dir, "convined_" + file_name + ".mp4"),
        )

    # # find train npy files
    # test_npy_list = glob.glob(os.path.join(test_pack_dir, "**.npy"))
    # print("find %d train bags!" % len(test_npy_list))
    # # train process
    # for npy_path in tqdm(test_npy_list, desc="train dataset"):
    #     file_name = os.path.splitext(os.path.basename(npy_path))[0]
    #     npy_dict = np.load(npy_path, allow_pickle=True).item()
    #     plot_observations(npy_dict, test_plot_dir, file_name)
    #     # TODO: the length of mp4 and wav is different
    #     convine_mp4_wav(
    #         os.path.join(test_plot_dir, file_name + ".mp4"),
    #         os.path.join(test_bag_dir, file_name, file_name + ".wav"),
    #         os.path.join(test_plot_dir, "convined_" + file_name + ".mp4"),
    #     )


if __name__ == "__main__":
    main()
