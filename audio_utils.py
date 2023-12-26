import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchaudio
import math
import numpy as np
import warnings
import soundfile as sf


def to_np(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return data


def to_tensor(data, dtype=np.float32):
    if isinstance(data, (np.ndarray, np.generic)):
        return torch.from_numpy(data.astype(dtype))
    return data


def mlsp_plot(mlsp, plot_dir, sr, hop_length) -> None:
    fig, ax = plt.subplots(figsize=(25, 5))
    img = librosa.display.specshow(mlsp, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    plt.savefig(plot_dir)


def mlsp_reshape(mlsp):
    # length, freq, frame = mlsp.shape
    mlsp = to_np(mlsp)
    result = np.concatenate([mlsp[i] for i in range(len(mlsp))], axis=1)
    return result


def save_wav(wav, file_path, sr):
    sf.write(file_path, wav, sr)

def waveform_normalize(wav):
    wav = to_np(wav)
    max = np.max(np.abs(wav))
    return np.multiply(wav, 1/max)

def sound_preprocess(wav_path, plot_path=None, divide_mode="mlsp", library="librosa"):
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
            if plot_path != None:
                mlsp_plot(mlsp, plot_path, sr=sr)
            # slice mlsp to target Hz
            freq, total_frame = mlsp.shape
            for i in range(int(total_frame // frame_num)):
                temp = mlsp.T[frame_num * i : frame_num * (i + 1)]
                sound_array.append(temp.T)
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            sound_array = np.array(sound_array).astype(np.float32)
            sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
        elif library == "torchaudio":
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
            temp = torch.FloatTensor(wav)
            # mlsp = power2db(trans_mel(temp))
            mlsp_power = trans_mel(temp)
            ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
            mlsp = torchaudio.functional.amplitude_to_DB(trans_mel(temp), multiplier, amin, math.log10(max(amin, ref_value)), top_db)
            if plot_path != None:
                temp = mlsp.to("cpu").detach().numpy().copy()
                mlsp_plot(temp, plot_path, sr=sr)
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            mlsp = mlsp.abs().float().div_(80).to("cpu").detach().numpy().copy()
            # slice mlsp to target Hz
            freq, total_frame = mlsp.shape
            for i in range(int(total_frame // frame_num)):
                temp = mlsp.T[frame_num * i : frame_num * (i + 1)]
                sound_array.append(temp.T)
        else:
            print("Error : please select library torchaudio or librosa", file=sys.stderr)
            raise NotImplementedError()
    return sound_array


def sound_postprocess(mlsp, library="librosa", device="cpu"):
    """
    convert from mel-spectrogram to waveform

    Args:
        mlsp (_type_): mel-spectrogram (value : 0~1)
        library (str, optional): select librosa or torchaudio. Defaults to "librosa".
        device (str, optional): you can select device when you select library torchaudio. Defaults to "cpu".

    Returns:
        npy_array: waveform
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
    
    if library=="librosa" and device in "cuda":
        warnings.warn("Warning : librosa work on CPU only. if you want to use GPU, you must select library torchaudio")
    device = torch.device(device)

    # reshape to array (length, freq)
    if len(mlsp.shape) == 3:
        mlsp = mlsp_reshape(mlsp)
    # convert to db [0 ~ -80]
    mlsp = np.multiply(to_np(mlsp), -80)
    
    if library == "librosa":
        mlsp = librosa.db_to_power(mlsp)
        wav = librosa.feature.inverse.mel_to_audio(mlsp, sr=sr, n_fft=fft_size, hop_length=hop_length)
    elif library == "torchaudio":
        inverseMel = torchaudio.transforms.InverseMelScale(
            n_stft=int(fft_size // 2 + 1),
            n_mels=n_mels,
            sample_rate=sr,
            f_min=0.0,
            f_max=sr / 2.0,
            max_iter=100000000,
            tolerance_loss=1e-50,
            tolerance_change=1e-08,
            sgdargs=None,
            norm=None,
            mel_scale="slaney",
        ).to(device=device)
        inverseWav = torchaudio.transforms.GriffinLim(
            n_fft=fft_size,
            n_iter=32,
            win_length=fft_size,
            hop_length=hop_length,
            power=2.0,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(device=device)
        
        mlsp = to_tensor(mlsp).to(device=device)
        mlsp = torchaudio.functional.DB_to_amplitude(mlsp, ref=1.0, power=1.0)
        spec = inverseMel(mlsp)
        wav = inverseWav(spec)
        wav = to_np(wav)
        del inverseMel, inverseWav, mlsp, spec
    else:
        print("Error : please select library torchaudio or librosa", file=sys.stderr)
        raise NotImplementedError()
    return wav