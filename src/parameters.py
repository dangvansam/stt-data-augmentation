import os


class Params:
    PWD: str = os.path.dirname(os.path.realpath(__file__))
    AUG_PATH: str = 'data/augmentation'
    NOISE_PATH: dict = {
        'noise': 'data/raw/test/noise/noise'
    }
    RATIO_VALID: float = 0.005


class FeatureParams:
    samplerate: int = 16000
    # noise Injector params
    noise_factor: float = 0.05
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.75  # Maximum noise levels to sample from. Maximum 1.0
    # Changing Pitch
    max_steps: int = 5
    # changing speed
    min_stretch: float = 0.8
    max_stretch: float = 1.2
    # change volume
    max_increase: int = 8
    max_decrease: int = -14
    # general prob
    prob: float = 1
    # gaussain noise
    min_amplitude: float = 0.02
    max_amplitude: float = 0.025
    # time
    min_rate: float = 0.8
    max_rate: float = 1.25
    # silence
    max_start_silence_secs: float = 0.5
    max_end_silence_secs: float = 0.5
    # gain
    min_gain_dbfs: float = -10
    max_gain_dbfs: float = 10
    # white noise
    min_level = -50
    max_level = -30
