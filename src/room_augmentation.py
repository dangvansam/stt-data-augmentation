from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy as np
import random
import pyroomacoustics


def make_room(
    wav_signal: np.array,
    room_x: dict = {'start': 5, 'stop': 21},
    room_y: dict = {'start': 5, 'stop': 21},
    room_z: dict = {'start': 3, 'stop': 11},
    wav_samplerate: int = 16000,
    max_order: int = 15,
    min_absorption: float = 0.1,
    max_absorption: float = 0.9,
    num_mics: int = 1,
    min_humidity: float = 40,
    max_humidity: float = 90,
    air_absorption: bool = False
) -> pyroomacoustics.ShoeBox:

    x = random.randrange(**room_x)
    y = random.randrange(**room_y)
    z = random.randrange(**room_z)

    room_dim = [x, y, z]
    source_location = [random.randrange(1, x - 1), random.randrange(1, y - 1), random.randrange(1, z - 1)]
    mic_location = source_location

    absorption = np.random.uniform(min_absorption, max_absorption)
    humidity = np.random.uniform(min_humidity, max_humidity)

    # Create the shoebox
    shoebox = pyroomacoustics.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=wav_samplerate,
        max_order=max_order,
        humidity=humidity,
        air_absorption=air_absorption
    )

    shoebox.add_source(source_location, signal=wav_signal)
    for i in range(num_mics):
        while mic_location == source_location:
            mic_location = [random.randrange(1, x), random.randrange(1, y), random.randrange(1, z)]
        shoebox.add_microphone_array(
            pyroomacoustics.MicrophoneArray(
                np.array([mic_location]).T,
                shoebox.fs
            )
        )
        mic_location = source_location
    shoebox.simulate()
    return shoebox


class SIM_ImpulseResponse(BaseWaveformTransform):
    """Simulation Impulse Responses"""

    def __init__(self, min_order=10, max_order=20, p=0.5):
        super().__init__(p)
        assert min_order > 1
        assert max_order < 30
        assert min_order <= max_order
        self.min_order = min_order
        self.max_order = max_order

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            """
            If rate > 1, then the signal is sped up.
            If rate < 1, then the signal is slowed down.
            """
            self.parameters["rate"] = random.randint(self.min_order, self.max_order)

    def apply(self, samples, sample_rate):
        try:
            shoebox = make_room(samples, max_order=self.max_order, wav_samplerate=sample_rate)
            waveform = shoebox.mic_array.signals[0]
        except Exception as ex:
            print(ex)
            waveform = samples
        return waveform