import os
import cv2
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchaudio
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image


# ------------------------------------------------------------ #
#                           image                              #
# ------------------------------------------------------------ #
def image_process(dataset_dir, folder_name, image_dir):
    image_list = glob.glob(os.path.join(dataset_dir, folder_name, image_dir, "*.jpg"))
    image_name_base = os.path.join(dataset_dir, folder_name, image_dir, "frame{:0>4d}.jpg")
    image_name_list = [image_name_base.format(i) for i in range(len(image_list))]
    image_array = []
    for image_name in image_name_list:
        im = Image.open(image_name)
        im_list = np.asarray(im)
        if image_dir == "camera_side_color":
            # _im = im_list[:, :480]
            _im = im_list[:, 80:560]
            _im = cv2.resize(_im, (80, 80))
        elif image_dir == "camera_top_color":
            # _im = im_list
            _im = im_list[280:480, 300:500]
            _im = cv2.resize(_im, (80, 80))
        image_array.append(_im)
    image_array = np.array(image_array)
    return image_array


def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


# ------------------------------------------------------------ #
#                           sound                              #
# ------------------------------------------------------------ #
def sound_process(wav_path, divide_mode="wav", library="librosa"):
    """
    wav file convert to mel-spectrogram numpy array divide 10Hz
    Args:
        wav_path (str): path to wav file
        plot_path (str, optional): if you want to plot mlsp, enter the path to save image dir. Defaults to None.
        divide_mode (str, optional): select wav or mlsp. Defaults to "wav".
        library (str, optional): select librosa or torchaudio. Defaults to "librosa".
    Returns:
        numpy array: 10Hz mlsp (normalized 0~1)
    """
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    n_mels = 128
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))
    top_db = 80.0
    multiplier = 10.0
    amin = 1e-10
    ref_value = np.max
    # db_multiplier = math.log10(max(amin, ref_value))

    if library == "torchaudio":
        trans_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=fft_size,
            win_length=None,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="slaney",
        )

    # create mel-spectrogram
    wav, sr = librosa.load(wav_path, sr=sr)
    sound_array = []
    if divide_mode == "wav":
        wav_frame_num = np.int64(sr * (1 / target_hz))
        if library == "librosa":
            for i in range(len(wav) // wav_frame_num):
                temp = wav[wav_frame_num * i : wav_frame_num * (i + 1)]
                mlsp = librosa.feature.melspectrogram(y=temp, sr=sr, n_fft=fft_size, hop_length=hop_length, htk=False)
                mlsp = librosa.power_to_db(mlsp, ref=ref_value)
                sound_array.append(mlsp[:, :frame_num])
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            sound_array = np.array(sound_array).astype(np.float32)
            sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
        elif library == "torchaudio":
            for i in range(len(wav) // wav_frame_num):
                temp = torch.FloatTensor(wav[wav_frame_num * i : wav_frame_num * (i + 1)])
                mlsp_power = trans_mel(temp)
                ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
                mlsp = torchaudio.functional.amplitude_to_DB(trans_mel(temp), multiplier, amin, math.log10(max(amin, ref_value)), top_db)
                # sound preprocess [-0 ~ -80] -> [0 ~ 1]
                mlsp = torch.narrow(mlsp.abs().float().div_(80), 1, 0, frame_num).to("cpu").detach().numpy().copy()
                sound_array.append(mlsp)
    elif divide_mode == "mlsp":
        if library == "librosa":
            mlsp = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=fft_size, hop_length=hop_length, htk=False)
            mlsp = librosa.power_to_db(mlsp, ref=ref_value)
            # slice mlsp to target Hz
            freq, total_frame = mlsp.shape
            for i in range(int(total_frame // frame_num)):
                temp = mlsp.T[frame_num * i : frame_num * (i + 1)]
                sound_array.append(temp.T)
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            sound_array = np.array(sound_array).astype(np.float32)
            sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
        elif library == "torchaudio":
            temp = torch.FloatTensor(wav)
            # mlsp = power2db(trans_mel(temp))
            mlsp_power = trans_mel(temp)
            ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
            mlsp = torchaudio.functional.amplitude_to_DB(trans_mel(temp), multiplier, amin, math.log10(max(amin, ref_value)), top_db)
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            mlsp = mlsp.abs().float().div_(80).to("cpu").detach().numpy().copy()
            # slice mlsp to target Hz
            freq, total_frame = mlsp.shape
            for i in range(int(total_frame // frame_num)):
                temp = mlsp.T[frame_num * i : frame_num * (i + 1)]
                sound_array.append(temp.T)
    return sound_array


def sound2mlsp(sound):
    if torch.is_tensor(sound):
        sound = tensor2numpy(sound)
    sound_mel = np.multiply(sound, -80)
    return sound_mel


def mlsp2wav(sound, sr, fft_size, hop_length):
    if torch.is_tensor(sound):
        sound = tensor2numpy(sound)
    sound_mel_db = np.multiply(sound, -80)
    sound_mel_pw = librosa.db_to_power(sound_mel_db)
    sound_wav = librosa.feature.inverse.mel_to_audio(sound_mel_pw, sr=sr, n_fft=fft_size, hop_length=hop_length)
    return sound_wav


# ------------------------------------------------------------ #
#                           csv file                           #
# ------------------------------------------------------------ #
def get_nearest_index(df, name, num):
    index = df.index[(df[name] - num).abs().argsort()][0].tolist()
    return index


def floor(x, digit=0):
    return np.floor(x * 10 ** digit) / (10 ** digit)


def slice_time(df, dt=0.1):
    df["dt[s]"] = (df["%time"] - df.iloc[0, 0]) / 1e9

    t_max = df["dt[s]"].max()

    indexes = []
    for t in np.arange(0, t_max, dt):
        index = get_nearest_index(df, "dt[s]", t)
        indexes.append(index)
    return df.loc[indexes, :]


def load_joint_states(path):
    df = pd.read_csv(path, header=None, skiprows=1, names=[i for i in range(28)])
    df2 = df.dropna(subset=[6])
    df3 = df2[[0, 2, 10, 11, 12, 13, 14, 15]]
    df4 = df3.rename(
        columns={
            0: "%time",
            2: "field.header.stamp",
            10: "joint_1",
            11: "joint_2",
            12: "joint_3",
            13: "joint_4",
            14: "joint_5",
            15: "joint_6",
        }
    )
    df4 = df4.astype(
        {
            "%time": "int64",
            "field.header.stamp": "int64",
            "joint_1": "float32",
            "joint_2": "float32",
            "joint_3": "float32",
            "joint_4": "float32",
            "joint_5": "float32",
            "joint_6": "float32",
        }
    )
    return df4


def get_joint_state(dataset_name, folder_name):
    joint_states = "joint_states.cvs"
    joint_csv = "{}/{}/{}".format(dataset_name, folder_name, joint_states)
    joint_raw = load_joint_states(joint_csv)
    joint_slice = slice_time(joint_raw, 0.1)
    joint_states = joint_slice.iloc[:, 2 : 2 + 6].values
    d_joint_states = joint_states[1:, :] - joint_states[:-1, :]
    return joint_states, d_joint_states


def get_ee_state(dataset_dir, folder_name):
    file_name = "ee_state.cvs"
    df_dict = {
        "%time": "int64",
        "field.header.stamp": "int64",
        "field.vector.x": "float32",
        "field.vector.y": "float32",
        "field.vector.z": "float32",
    }

    cvs_path = os.path.join(dataset_dir, folder_name, file_name)
    df = pd.read_csv(cvs_path)
    df = df[df_dict.keys()]
    df = df.astype(df_dict)
    df = slice_time(df, dt=0.1)
    ee_states = df.iloc[:, 2 : 2 + 3].values
    d_ee_states = ee_states[1:, :] - ee_states[:-1, :]
    return ee_states, d_ee_states


def get_ss_twist(dataset_dir, folder_name):
    file_name = "servo_twist.cvs"
    df_dict = {
        "%time": "int64",
        "field.header.stamp": "int64",
        "field.twist.linear.x": "float32",
        "field.twist.linear.y": "float32",
        "field.twist.linear.z": "float32",
    }

    cvs_path = os.path.join(dataset_dir, folder_name, file_name)
    df = pd.read_csv(cvs_path)
    df = df[df_dict.keys()]
    df = df.astype(df_dict)
    df = slice_time(df, dt=0.1)
    ss_twist = df.iloc[:, 2 : 2 + 3].values
    return ss_twist


# ------------------------------------------------------------ #
#                           others                             #
# ------------------------------------------------------------ #
def tensor2numpy(tensor):
    return tensor.to("cpu").detach().numpy().copy()


def make_dummy(data_length):
    action = np.zeros((data_length, 1)).astype(np.float32)
    reward = np.zeros((data_length,)).astype(np.float32)
    done = np.zeros((data_length,)).astype(np.float32)
    done[-1] = 1.0
    return action, reward, done


def sync_data(data: dict) -> dict:
    index_list = []
    for key in data.keys():
        index_list.append(len(data[key]))
    min_index = np.min(index_list)
    for key in data.keys():
        data[key] = data[key][:min_index]
    action, reward, done = make_dummy(data_length=min_index)
    data["action"] = action
    data["reward"] = reward
    data["done"] = done
    return
