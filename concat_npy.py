import os
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

from utils.file_reader import *


def tensor2numpy(array):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy().copy()
    return array


def get_latent_state(data, key):
    if key == "beliefs":
        temp = tensor2numpy(data[key][0])
        for i in range(1, len(data[key])):
            temp = np.concatenate([temp, np.squeeze(tensor2numpy(data[key][i]), axis=0)], axis=0)
        temp = np.expand_dims(temp, axis=0)
    elif key == "prior_states":
        temp = tensor2numpy(data[key][1])
    elif key == "posterior_states":
        temp = tensor2numpy(data[key][0])
        temp = np.expand_dims(np.expand_dims(temp, axis=0), axis=0)
    return temp


def main():
    numpy_path = "numpy/2022-03-24"
    numpy_dir_list = os.listdir(numpy_path)

    results_path = "result/2022-03-24"
    os.makedirs(results_path, exist_ok=True)

    for numpy_dir in numpy_dir_list:
        start_time = 0
        stamps = {}
        observations = {}
        latent_state = {}
        latent_state_keys = ["beliefs", "prior_states", "posterior_states"]

        dir_path = os.path.join(numpy_path, numpy_dir)
        n_frame = sum(os.path.isfile(os.path.join(dir_path, name)) for name in os.listdir(dir_path))
        for i in range(n_frame):
            file_path = os.path.join(dir_path, "frame_%05d.npy" % i)
            dict = np.load(file_path, allow_pickle=True).item()
            # 'stamp_up', 'stamp_action', 'observations', 'action', 'beliefs', 'prior_states', 'posterior_states'
            # time stamps and action
            if i == 0:
                start_time = dict["stamp_up"]
                stamps["uptime"] = np.array(dict["stamp_up"] - start_time)
                stamps["latency"] = np.array(dict["stamp_action"] - dict["stamp_up"])
                actions = np.expand_dims(dict["action"], axis=0)
                # RSSM parameters
                for key in latent_state_keys:
                    latent_state[key] = get_latent_state(dict, key)
            else:
                stamps["uptime"] = np.append(stamps["uptime"], dict["stamp_up"] - start_time)
                stamps["latency"] = np.append(stamps["uptime"], dict["stamp_action"] - dict["stamp_up"])
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
        for key in latent_state_keys:
            print(key + " : " + str(latent_state[key].shape))


if __name__ == "__main__":
    main()
