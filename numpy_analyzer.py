import os
from turtle import color
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import librosa.display
import torch
import torchaudio
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import pickle as pk


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


def pca_process(data, key, pca_model):
    shape = data[key].shape[:-1] + (3,)
    result = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        result[i] = pca_model.transform(data[key][i])
    return result


def get_latent_state(data, key):
    if key == "beliefs":
        temp = tensor2numpy(data[key][0])
        for i in range(1, len(data[key])):
            temp = np.concatenate([temp, np.squeeze(tensor2numpy(data[key][i]), axis=0)], axis=0)
        temp = np.expand_dims(temp, axis=0)
    elif key == "prior_states":
        temp = np.expand_dims(np.expand_dims(tensor2numpy(data["posterior_states"][0]), axis=0), axis=0)
        temp = np.concatenate([temp, tensor2numpy(data[key][1])], axis=1)
    return temp


def numpy_loader(numpy_dir, numpy_path):
    start_time = 0
    stamps = {}
    observations = {}
    latent_state = {}
    latent_state_keys = ["beliefs", "prior_states"]

    dir_path = os.path.join(numpy_path, numpy_dir)
    n_frame = sum(os.path.isfile(os.path.join(dir_path, name)) for name in os.listdir(dir_path))
    for i in tqdm(range(n_frame), desc="loading", leave=False):
        file_path = os.path.join(dir_path, "frame_%05d.npy" % i)
        dict = np.load(file_path, allow_pickle=True).item()
        # time stamps and action
        if i == 0:
            start_time = dict["stamp_up"]
            stamps["uptime"] = np.array(dict["stamp_up"] - start_time, dtype=np.float32)
            stamps["latency"] = np.array(dict["stamp_action"] - dict["stamp_up"], dtype=np.float32)
            actions = np.expand_dims(dict["action"], axis=0)
            # RSSM parameters
            for key in latent_state_keys:
                latent_state[key] = get_latent_state(dict, key)
        else:
            stamps["uptime"] = np.append(stamps["uptime"], dict["stamp_up"] - start_time).astype(np.float32)
            stamps["latency"] = np.append(stamps["latency"], dict["stamp_action"] - dict["stamp_up"]).astype(np.float32)
            actions = np.concatenate([actions, np.expand_dims(dict["action"], axis=0)], axis=0)
            for key in latent_state_keys:
                latent_state[key] = np.concatenate([latent_state[key], get_latent_state(dict, key)], axis=0)
        # observation
        observation = dict["observations"]
        for obs_key in observation.keys():
            if not obs_key in observations.keys():
                observations[obs_key] = tensor2numpy(observation[obs_key])
            else:
                observations[obs_key] = np.concatenate([observations[obs_key], tensor2numpy(observation[obs_key])], axis=0)

    # for key in stamps.keys():
    #     print(key + " : " + str(stamps[key].shape))
    # for key in observations.keys():
    #     print(key + " : " + str(observations[key].shape))
    # for key in latent_state_keys:
    #     print(key + " : " + str(latent_state[key].shape))

    return stamps, observations, actions, latent_state


# def plot_ls_traj(ax, traj_rec, traj_imag, t, dim=2):
#     x_rec, y_rec, z_rec = get_xyz(traj_rec[: t + 1])
#     x_imag, y_imag, z_imag = get_xyz(traj_imag[: t + 1])
#     if dim == 3:
#         ax.plot(x_rec, y_rec, z_rec, label="rec", marker="x")
#         ax.plot(x_imag, y_imag, z_imag, label="imag", marker="x")
#     else:
#         ax.plot(x_rec, y_rec, label="rec", marker="x")
#         ax.plot(x_imag, y_imag, label="imag", marker="x")
#     ax.set_xlim(-10, 10)
#     ax.set_ylim(-10, 10)
#     # ax.set_xlabel("X-axis")
#     # ax.set_ylabel("Y-axis")
#     if dim == 3:
#         ax.set_zlim(-10, 10)
#         # ax.set_zlabel("Z-axis")


def plot_ls_traj(data, ax, t, dim=2):
    x_rec, y_rec, z_rec = get_xyz(data[:t, 0])
    x_imag, y_imag, z_imag = get_xyz(data[t, :])
    if dim == 3:
        ax.plot(x_rec, y_rec, z_rec, marker="x")
        ax.plot(x_imag, y_imag, z_imag, marker="x")
    else:
        ax.plot(x_rec, y_rec, marker="x")
        ax.plot(x_imag, y_imag, marker="x")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    if dim == 3:
        ax.set_zlim(-5, 5)
        # ax.set_zlabel("Z-axis")


def plot_eepos_traj(data, ax, t, dim=3):
    pos_x, pos_y, pos_z = get_xyz(data[:t])
    if dim == 3:
        ax.plot(pos_x, pos_y, pos_z, marker="x")
    else:
        ax.plot(pos_x, pos_y, marker="x")
    ax.set_xlim(0.2, 0.3)
    ax.set_ylim(0.005, 0.01)
    if dim == 3:
        ax.set_zlim(0.15, 0.25)


def plot_ts_line(data, x_key, y_key, ax, t, length=10):
    if t < length:
        data_x = data[x_key][: t + 1]
        data_y = data[y_key][: t + 1]
    else:
        data_x = data[x_key][t + 1 - length : t + 1]
        data_y = data[y_key][t + 1 - length : t + 1]
    ax.plot(data_x, data_y, color="k", marker = "x")
    ax.set_xticks(np.arange(min(data_x) - 0.1, max(data_x) + 0.1, 0.1))
    ax.set_ylim(min(data[y_key]) - 0.01, max(data[y_key]) + 0.01)


def plot_action_bar(actions, ax, t):
    height = actions[t].flatten()
    x_axis = np.array([1, 2, 3])
    label = ["X", "Y", "Z"]
    ax.bar(x_axis, height, tick_label=label, align="center", color="skyblue")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylim(np.min(actions) - 0.005, np.max(actions) + 0.005)


def plot_stamp_obs_action_ls(stamps, observations, actions, latent_state, pca_belief, pca_post_mean, plot_dir, file_name, n_frame=None, pca_dim=2):
    if n_frame == None:
        n_frame = len(observations["image_horizon"])

    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    clip_belief = pca_process(latent_state, "beliefs", pca_belief)
    clip_prior_states = pca_process(latent_state, "prior_states", pca_post_mean)

    fig = plt.figure(figsize=(5 * 4, 5 * 2))
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    if pca_dim == 2:
        ax3 = fig.add_subplot(2, 4, 3)
    elif pca_dim == 3:
        ax3 = fig.add_subplot(2, 4, 3, projection="3d")
    ax4 = fig.add_subplot(2, 4, 4)
    ax5 = fig.add_subplot(2, 4, 5, projection="3d")
    # ax6 = fig.add_subplot(2, 4, 6)
    if pca_dim == 2:
        ax7 = fig.add_subplot(2, 4, 7)
    elif pca_dim == 3:
        ax7 = fig.add_subplot(2, 4, 7, projection="3d")
    ax8 = fig.add_subplot(2, 4, 8)

    def plot(t):
        plt.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        # ax6.cla()
        ax7.cla()
        ax8.cla()

        # Observation
        imh = postprocess_observation(observations["image_horizon"][t])
        ax1.imshow(imh.transpose(1, 2, 0)[:, :, ::-1])
        mlsp = sound_postprocess(observations["sound"][t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax2)
        plot_eepos_traj(observations["end_effector"], ax5, t)

        # latent state
        plot_ls_traj(clip_belief, ax=ax3, t=t, dim=pca_dim)
        plot_ls_traj(clip_prior_states, ax=ax7, t=t, dim=pca_dim)

        # action
        plot_action_bar(actions, ax=ax4, t=t)
        # stamp
        plot_ts_line(stamps, "uptime", "latency", ax=ax8, t=t, length=5)

        fig.suptitle("observation t={}".format(t))
        ax1.title.set_text("Horizontal Image")
        ax2.title.set_text("Mel-Spectrogram")
        ax3.title.set_text("Deterministic")
        ax4.title.set_text("Action")
        ax5.title.set_text("End-effector Position")
        # ax6.title.set_text("")
        ax7.title.set_text("Stochastic")
        ax8.title.set_text("Latency")

    # create animation 10Hz
    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_plot_name = os.path.join(plot_dir, file_name + "_" + str(pca_dim) + "d" + ".mp4")
    anim.save(save_plot_name, writer="ffmpeg")


def main():
    n_frame = None
    
    numpy_path = "numpy/2022-03-24"
    results_path = "result/2022-03-24"
    pca_belief = pk.load(open("pca/pca_belief.pkl", "rb"))
    pca_post_mean = pk.load(open("pca/pca_post_mean.pkl", "rb"))

    os.makedirs(results_path, exist_ok=True)

    numpy_dir_list = os.listdir(numpy_path)
    for numpy_dir in tqdm(numpy_dir_list, desc="numpy load process"):
        stamps, observations, actions, latent_state = numpy_loader(numpy_dir, numpy_path)
        plot_stamp_obs_action_ls(stamps, observations, actions, latent_state, pca_belief, pca_post_mean, results_path, numpy_dir, n_frame=n_frame, pca_dim=2)
        plot_stamp_obs_action_ls(stamps, observations, actions, latent_state, pca_belief, pca_post_mean, results_path, numpy_dir, n_frame=n_frame, pca_dim=3)


if __name__ == "__main__":
    main()
