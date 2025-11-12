from pydub import AudioSegment
from audiomentations import (
    AddGaussianNoise,
    AddShortNoises,
    Normalize,
    Mp3Compression,
    ApplyImpulseResponse,
)
from room_augmentation import make_room
import numpy as np
import soundfile as sf
import scipy
import librosa
import random
import subprocess
import os
from joblib import Parallel, delayed
from typing import List
from parameters import Params, FeatureParams
from detect_non_sil import detect_non_silence

MAX_SNR = 15

def load_noise(directory: str) -> List[str]:
    list_paths = []
    for f in os.listdir(directory):
        noise_path = os.path.join(directory, f)
        list_paths.append(noise_path)
    return list_paths

def load_data(dataset: str) -> List[str]:
    return [("path", "text", "duration")]


class DataAugmentation:
    '''
    Augmentation audio with change speed, pitch, volumne and
    add background noise follow signal-to-noise ratio

    '''

    def __init__(
        self,
        params: Params,
        feat_params: FeatureParams,
        mode: str = 'noise',
        pwd: str = None,
        noise_path: str = None,
    ) -> None:

        self.params = params
        self.feat_params = feat_params
        self.list_noise_paths = load_noise(os.path.join(pwd, noise_path))
        print(f'Length nosie path {len(self.list_noise_paths)}')
        if mode == "gaussnoise":
            self.gauss_aug = AddGaussianNoise(
                min_amplitude=feat_params.min_amplitude,
                max_amplitude=feat_params.max_amplitude,
                p=feat_params.prob
            )
        elif mode == "shortnoise":
            self.snoise_aug = AddShortNoises(sounds_path=os.path.join(pwd, noise_path), p=feat_params.prob)
        elif mode == "norm":
            self.norm = Normalize(p=feat_params.prob)
        elif mode == "mp3_compress":
            self.mp3_compress = Mp3Compression(p=feat_params.prob)
        elif mode == "rir":
            self.rir = ApplyImpulseResponse(ir_path=os.path.join(pwd, noise_path), p=feat_params.prob)

    def _short_noise(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return self.snoise_aug(signal, sr)

    def _gaussian_noise(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return self.gauss_aug(signal, sr)

    def _room(self, signal: np.ndarray, sr: int, audio_path: str) -> np.ndarray:
        shoebox = make_room(signal, wav_samplerate=sr)
        _ = shoebox.mic_array.to_wav(audio_path, norm=True, bitdepth=np.float32)  # 1D numpy array
        signal, _ = sf.read(audio_path)
        return signal

    def _down_sr(self, filepath):
        filename = filepath.split("/")[-1]
        tmp_file = os.path.join(f"tmp/{filename}")
        cmd = f"ffmpeg -hide_banner -loglevel error -i {filepath} -ar 8000 {tmp_file}".split()
        # try:
        subprocess.Popen(cmd).wait()
        # except:
            # return None
        audio, sr = librosa.load(tmp_file, sr=16000)
        os.remove(tmp_file)
        return audio, sr

    def _change_pitch(self, signal: np.ndarray) -> np.ndarray:
        step = np.random.uniform(
            low=-self.feat_params.max_steps,
            high=self.feat_params.max_steps)

        signal = librosa.effects.pitch_shift(
            signal,
            sr=self.feat_params.samplerate,
            n_steps=step
        )

        return signal

    def _change_speed(self, signal: np.ndarray) -> np.ndarray:
        speed_factor = np.random.uniform(self.feat_params.min_stretch, self.feat_params.max_stretch)
        signal = librosa.effects.time_stretch(signal, speed_factor)

        return signal

    def _add_noise(self, audio_path: str) -> AudioSegment:
        noise_path = random.choice(self.list_noise_paths)
        noise = AudioSegment.from_file(noise_path)
        sound = AudioSegment.from_file(audio_path)

        sound_power, noise_power = sound.dBFS, noise.dBFS
        temp = np.random.uniform(0, MAX_SNR)
        change_dB = sound_power - temp - noise_power

        noise = noise + change_dB

        combined = sound.overlay(noise, loop=True)

        return combined

    def _change_volume(self, audio_path: str) -> AudioSegment:
        sound = AudioSegment.from_file(audio_path)
        dB = np.random.uniform(
            low=self.feat_params.max_decrease,
            high=self.feat_params.max_increase
        )
        sound = sound + dB

        return sound

    def _convolve_rir(self, audio_path: str) -> np.ndarray:
        rir_path = random.choice(self.list_noise_paths)
        speech, _ = sf.read(audio_path)
        power = (speech[detect_non_silence(speech)] ** 2).mean()
        rir, _ = sf.read(rir_path, dtype=np.float64, always_2d=False)

        # rir: (Nmic, Time)
        rir = rir.T
        # speech: (Nmic, Time)
        # Note that this operation doesn't change the signal length
        speech = scipy.signal.convolve(speech, rir, mode="full")
        # Reverse mean power to the original power
        power2 = (speech[detect_non_silence(speech)] ** 2).mean()
        speech = np.sqrt(power / max(power2, 1e-10)) * speech

        return speech

    def _add_sil(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """ Applies random silence at the start and/or end of the audio."""

        start_silence_len = random.uniform(0, self.feat_params.max_start_silence_secs)
        end_silence_len = random.uniform(0, self.feat_params.max_end_silence_secs)
        start = np.full((int(start_silence_len * sr),), 0.0)
        end = np.full((int(end_silence_len * sr),), 0.0)
        signal = np.concatenate([start, signal, end])

        return signal

    def _gain(self, signal: np.ndarray) -> np.ndarray:
        """Applies random gain to the audio."""

        gain = random.uniform(self.feat_params.min_gain_dbfs, self.feat_params.max_gain_dbfs)
        signal = signal * (10.0 ** (gain / 20.0))

        return signal

    def _white_noise(self, signal: np.ndarray) -> np.ndarray:
        """Perturbation that adds white noise to an audio file in the training dataset."""

        noise_level_db = np.random.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = np.random.randn(signal.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        signal += noise_signal

        return signal

    def _norm(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return self.norm(signal, sr)

    def _mp3_compress(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return self.mp3_compress(signal, sr)

    def _rir(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return self.rir(signal, sr)


def _process_aug(audio_path: str, text: str, duration: str, mode: str):
    utt = audio_path.split('/')[-1].replace('.wav', '')    # wav, flac
    audio_aug_path = os.path.join(aug_wav_dir, f'{utt}.{mode}.wav')
    if not os.path.exists(audio_aug_path) or os.path.getsize(audio_aug_path) <= 10000:
        try:
            if os.path.exists(audio_path):
                signal, sr = librosa.load(audio_path, sr=DA.feat_params.samplerate)
                if mode == 'down_sr':
                    aug_signal, sr = DA._down_sr(audio_path)
                elif mode not in ['gaussnoise', 'shortnoise'] and 'noise' in mode:
                    aug_signal = DA._add_noise(audio_path)
                elif mode == 'change_volume':
                    aug_signal = DA._change_volume(audio_path)
                elif mode == 'shift_pitch':
                    aug_signal = DA._change_pitch(signal)
                elif mode == 'change_speed':
                    aug_signal = DA._change_speed(signal)
                elif mode == 'gaussnoise':
                    aug_signal = DA.gauss_aug(signal, sr)
                elif mode == 'shortnoise':
                    aug_signal = DA.snoise_aug(signal, sr)
                elif mode == 'room':
                    aug_signal = DA._room(signal, sr, audio_aug_path)
                elif mode == 'rir':
                    aug_signal = DA._rir(signal, sr)
                elif mode == 'sil':
                    aug_signal = DA._add_sil(signal, sr)
                elif mode == 'gain':
                    aug_signal = DA._gain(signal)
                elif mode == 'white_noise':
                    aug_signal = DA._white_noise(signal)
                elif mode == 'norm':
                    aug_signal = DA._norm(signal, sr)
                elif mode == 'mp3_compress':
                    aug_signal = DA._mp3_compress(signal, sr)
                else:
                    raise ValueError(f"Not supported: {mode}")
            else:
                print(f"file not exist: {audio_path}")
            if aug_signal is None:
                return None
            if mode in ['change_volume'] or (mode not in ['shortnoise', 'gaussnoise'] and 'noise' in mode):
                # save with pydub
                aug_signal.export(audio_aug_path, format='wav', bitrate='256k', parameters=["-ac", "1", "-ar", "16000"])
            elif mode == 'room':
                pass
            else:
                # save with soundfile
                sf.write(audio_aug_path, aug_signal, samplerate=sr)

        except ValueError:
            return None

    temp = audio_aug_path.replace('../', '/')
    return f'{temp}|{text}|{duration.strip()}\n'


def main(dataset: str, mode: str, pwd: str = '') -> None:
    params = Params()
    feat_params = FeatureParams()

    global aug_wav_dir, DA

    nosie_path = params.NOISE_PATH.get(mode, params.NOISE_PATH['noise'])

    DA = DataAugmentation(
        params=params,
        feat_params=feat_params,
        mode=mode,
        pwd=pwd,
        noise_path=nosie_path,
    )

    aug_wav_dir = os.path.join('/e2e-automatic-speech-recognition/data/augmentation', mode, dataset, 'wavs')
    aug_dir = os.path.dirname(aug_wav_dir)

    if mode == "down_sr":
        os.makedirs("tmp", exist_ok=True)

    if not os.path.exists(aug_wav_dir):
        os.makedirs(aug_wav_dir)

    # load raw audio of specific dataset
    data = load_data(dataset)

    print(len(data))

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(_process_aug)(audio_path, text, duration, mode) for audio_path, text, duration in data
    )

    results = [r for r in results if r is not None]
    with open(os.path.join(aug_dir, 'transcripts.txt'), 'w', encoding='utf-8') as f:
        for r in results:
            if r is not None:
                f.write(r)

    if mode == "down_sr" and os.path.exists("tmp"):
        os.rmdir("tmp")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwd",
                        type=str,
                        default='.',
                        help='current work directory')
    parser.add_argument("--dataset",
                        type=str,
                        default='vivos',
                        help='select dataset')
    parser.add_argument("--aug",
                        type=str,
                        default='change_volume',
                        help='type augmentation audio datasets')

    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.aug, pwd=args.pwd)
