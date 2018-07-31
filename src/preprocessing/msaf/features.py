"""
Each feature must inherit from the base class :class:`msaf.base.Features` to be
included in the whole framework.

Here is a list of all the available features:

.. autosummary::
    :toctree: generated/

    CQT
    MFCC
    PCP
    Tonnetz
    Tempogram
    Features
"""

from builtins import super
import librosa
import numpy as np

# Local stuff
from msaf import config
from msaf.base import Features
from msaf.exceptions import FeatureParamsError

def to_chroma(pianoroll):
    num_octave = pianoroll.shape[1] // 12
    chroma = np.sum(np.reshape(pianoroll[:, :(12 * num_octave)],
                               (-1, 12, num_octave)), axis=2)
    remainder = pianoroll.shape[1] % 12
    if remainder:
        chroma[:, :remainder] += pianoroll[:, -remainder:]
    return chroma

def get_normalized(data):
    normalized = np.nan_to_num(data / data.sum(axis=1, keepdims=True))
    return normalized

class CQT(Features):
    """This class contains the implementation of the Constant-Q Transform.

    These features contain both harmonic and timbral content of the given
    audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, mode=config.cqt.mode,
                 norm=config.cqt.norm, epsilon=config.cqt.epsilon,
                 ref_power=config.cqt.ref_power):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        mode: {'stack', 'merge'}
            Mode for flattening multi-track piano-rolls.
        norm: bool
            True to normalize the piano-rolls. False to do nothing.
        epsilon: float
            A small number added to the results to avoid divide by zero errors.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the CQT parameters
        if mode not in ('stack', 'merge'):
            raise FeatureParamsError("Wrong value for mode")
        self.mode = mode
        self.norm = norm
        self.epsilon = epsilon
        if ref_power == "max":
            self.ref_power = np.max
        elif ref_power == "min":
            self.ref_power = np.min
        elif ref_power == "median":
            self.ref_power = np.median
        else:
            raise FeatureParamsError("Wrong value for ref_power")

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "cqt"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        cqt: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        self._audio.binarize()
        if self.mode == 'stack':
            fpianorolls = [track.pianoroll for track in self._audio.tracks]
            flattened = np.concatenate(pianorolls, axis=1).astype(float)
        elif self.mode == 'merge':
            flattened = self._audio.get_merged_pianoroll().astype(float)
        flattened += self.epsilon * np.random.normal(size=flattened.shape)
        if self.norm:
            flattened = get_normalized(flattened)
        flattened = librosa.amplitude_to_db(flattened, ref=self.ref_power)
        return flattened


class MFCC(Features):
    """This class contains the implementation of the MFCC Features.

    The Mel-Frequency Cepstral Coefficients contain timbral content of a
    given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, mode=config.mfcc.mode,
                 norm=config.mfcc.norm, epsilon=config.mfcc.epsilon,
                 ref_power=config.mfcc.ref_power):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        mode: {'stack', 'merge'}
            Mode for flattening multi-track piano-rolls.
        norm: bool
            True to normalize the piano-rolls. False to do nothing.
        epsilon: float
            A small number added to the results to avoid divide by zero errors.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the MFCC parameters
        if mode not in ('stack', 'merge'):
            raise FeatureParamsError("Wrong value for mode")
        self.mode = mode
        self.norm = norm
        self.epsilon = epsilon
        if ref_power == "max":
            self.ref_power = np.max
        elif ref_power == "min":
            self.ref_power = np.min
        elif ref_power == "median":
            self.ref_power = np.median
        else:
            raise FeatureParamsError("Wrong value for ref_power")

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "mfcc"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        mfcc: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        self._audio.binarize()
        if self.mode == 'stack':
            pianorolls = [track.pianoroll for track in self._audio.tracks]
            flattened = np.concatenate(pianorolls, axis=1).astype(float)
        elif self.mode == 'merge':
            flattened = self._audio.get_merged_pianoroll().astype(float)
        flattened += self.epsilon * np.random.normal(size=flattened.shape)
        if self.norm:
            flattened = get_normalized(flattened)
        flattened = librosa.amplitude_to_db(flattened, ref=self.ref_power)
        return flattened


class PCP(Features):
    """This class contains the implementation of the Pitch Class Profiles.

    The PCPs contain harmonic content of a given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, mode=config.pcp.mode,
                 norm=config.pcp.norm, epsilon=config.pcp.epsilon,
                 ref_power="max"):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        mode: {'stack', 'merge'}
            Mode for flattening multi-track piano-rolls.
        norm: bool
            True to normalize the piano-rolls. False to do nothing.
        epsilon: float
            A small number added to the results to avoid divide by zero errors.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the PCP parameters
        if mode not in ('stack', 'merge'):
            raise FeatureParamsError("Wrong value for mode")
        self.mode = mode
        self.norm = norm
        self.epsilon = epsilon
        if ref_power == "max":
            self.ref_power = np.max
        elif ref_power == "min":
            self.ref_power = np.min
        elif ref_power == "median":
            self.ref_power = np.median
        else:
            raise FeatureParamsError("Wrong value for ref_power")

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "pcp"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        pcp: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        audio_harmonic, _ = self.compute_HPSS()
        if self.mode == 'stack':
            chromas = [to_chroma(track.pianoroll)
                                     for track in self._audio.tracks]
            flattened = np.concatenate(chromas, axis=1).astype(float)
        elif self.mode == 'merge':
            flattened = to_chroma(self._audio.get_merged_pianoroll())
            flattened = flattened.astype(float)
        flattened += self.epsilon * np.random.normal(size=flattened.shape)
        if self.norm:
            flattened = get_normalized(flattened)
        flattened = librosa.amplitude_to_db(flattened, ref=self.ref_power)
        return flattened


class Tonnetz(Features):
    """This class contains the implementation of the Tonal Centroids.

    The Tonal Centroids (or Tonnetz) contain harmonic content of a given audio
    signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, n_bins=config.tonnetz.bins,
                 norm=config.tonnetz.norm, f_min=config.tonnetz.f_min,
                 n_octaves=config.tonnetz.n_octaves):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of bins for the CQT computation.
        norm: int > 0
            Normalization parameter.
        f_min: float > 0
            Minimum frequency.
        n_octaves: int > 0
            Number of octaves.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the local parameters
        self.n_bins = n_bins
        self.norm = norm
        self.f_min = f_min
        self.n_octaves = n_octaves

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "tonnetz"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        tonnetz: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        pcp = PCP(self.file_struct, self.feat_type, self.sr, self.hop_length,
                  self.n_bins, self.norm, self.f_min, self.n_octaves).features
        tonnetz = librosa.feature.tonnetz(chroma=pcp.T).T
        return tonnetz


class Tempogram(Features):
    """This class contains the implementation of the Tempogram feature.

    The Tempogram contains rhythmic content of a given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size,
                 win_length=config.tempogram.win_length):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        win_length: int > 0
            The size of the window for the tempogram.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the local parameters
        self.win_length = win_length

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "tempogram"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        tempogram: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        return librosa.feature.tempogram(self._audio, sr=self.sr,
                                         hop_length=self.hop_length,
                                         win_length=self.win_length).T
