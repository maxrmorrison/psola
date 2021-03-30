import functools
import os
import tempfile

import numpy as np
import pypar
import soundfile
import tqdm
from parselmouth import Data, praat, Sound


__all__ = ['FMAX',
           'FMIN',
           'from_file',
           'from_file_to_file',
           'from_files_to_files',
           'to_file',
           'vocode']


###############################################################################
# Constants
###############################################################################


FMIN = 40.
FMAX = 550.


###############################################################################
# Vocode
###############################################################################


def from_file(audio_file,
              source_alignment_file=None,
              target_alignment_file=None,
              target_pitch_file=None,
              constant_stretch=None,
              fmin=FMIN,
              fmax=FMAX):
    """Performs vocoding using Praat

    Arguments
        audio_file : string
            The file containing the speech signal to process
        source_alignment_file : string or None
            The file containing the original alignment
        target_alignment_file : string or None
            The file containing the target alignment
        target_pitch_file : string or None
            The file containing the target pitch
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.

    Returns
        audio : np.array(shape=(samples,))
            The vocoded audio
        sample_rate : int
            The audio sampling rate
    """
    # Load inputs from file
    audio, sample_rate = soundfile.read(audio_file)
    source_alignment = None if source_alignment_file is None \
        else pypar.Alignment(source_alignment_file)
    target_alignment = None if target_alignment_file is None \
        else pypar.Alignment(target_alignment_file)
    target_pitch = None if target_pitch_file is None \
        else np.load(target_pitch_file)

    # Vocode
    audio = vocode(audio,
                   sample_rate,
                   source_alignment,
                   target_alignment,
                   target_pitch,
                   constant_stretch,
                   fmin,
                   fmax)

    return audio, sample_rate


def from_file_to_file(audio_file,
                      output_file,
                      source_alignment_file=None,
                      target_alignment_file=None,
                      target_pitch_file=None,
                      constant_stretch=None,
                      fmin=FMIN,
                      fmax=FMAX):
    """Performs vocoding using Praat and save to disk

    Arguments
        audio_file : string
            The file containing the speech signal to process
        output_file : string
            The file to save the vocoded speech
        source_alignment_file : string or None
            The file containing the original alignment
        target_alignment_file : string or None
            The file containing the target alignment
        target_pitch_file : string or None
            The file containing the target pitch
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
    """
    # Load and vocode
    audio, sample_rate = from_file(audio_file,
                                   source_alignment_file,
                                   target_alignment_file,
                                   target_pitch_file,
                                   constant_stretch,
                                   fmin,
                                   fmax)

    # Save to disk
    save(output_file, audio, sample_rate)


def from_files_to_files(audio_files,
                        output_files,
                        source_alignment_files=None,
                        target_alignment_files=None,
                        target_pitch_files=None,
                        constant_stretch=None,
                        fmin=FMIN,
                        fmax=FMAX):
    """Performs vocoding using Praat and save to disk

    Arguments
        audio_files : list
            The files containing the speech signals to process
        output_files : list
            The files to save the vocoded speech
        source_alignment_files : string or None
            The files containing the original alignments
        target_alignment_files : list or None
            The files containing the target alignments
        target_pitch_files : list or None
            The files containing the target pitch
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
    """
    # Expand None-valued defaults
    if source_alignment_files is None:
        source_alignment_files = [None] * len(audio_files)
    if target_alignment_files is None:
        target_alignment_files = [None] * len(audio_files)
    if target_pitch_files is None:
        target_pitch_files = [None] * len(audio_files)

    # Bind static parameters
    vocode_fn = functools.partial(from_file_to_file,
                                  constant_stretch=constant_stretch,
                                  fmin=fmin,
                                  fmax=fmax)

    # Setup iterator
    iterator = zip(audio_files,
                   output_files,
                   source_alignment_files,
                   target_alignment_files,
                   target_pitch_files)
    iterator = tqdm.tqdm(iterator, desc='psola', dynamic_ncols=True)
    for item in iterator:

        # Vocode and save to disk
        vocode_fn(*item)


def to_file(audio,
            sample_rate,
            output_file,
            source_alignment=None,
            target_alignment=None,
            target_pitch=None,
            constant_stretch=None,
            fmin=FMIN,
            fmax=FMAX):
    """Performs pitch vocoding and saves audio to disk

    Arguments
        audio : np.array(shape=(samples,))
            The speech signal to process
        sample_rate : int
            The audio sampling rate
        output_file : string
            The file to save the vocoded speech
        source_alignment : pypar.Alignment
            The current alignment if performing time-stretching
        target_alignment : pypar.Alignment
            The target alignment if performing time-stretching
        target_pitch : np.array(shape=(frames,))
            The target pitch contour
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
    """
    vocoded = vocode(audio,
                     sample_rate,
                     source_alignment,
                     target_alignment,
                     target_pitch,
                     constant_stretch,
                     fmin,
                     fmax)

    # Save
    save(output_file, vocoded, sample_rate)


def vocode(audio,
           sample_rate,
           source_alignment=None,
           target_alignment=None,
           target_pitch=None,
           constant_stretch=None,
           fmin=FMIN,
           fmax=FMAX):
    """Performs pitch vocoding using Praat

    Arguments
        audio : np.array(shape=(samples,))
            The speech signal to process
        sample_rate : int
            The audio sampling rate.
        source_alignment : pypar.Alignment
            The current alignment if performing time-stretching
        target_alignment : pypar.Alignment
            The target alignment if performing time-stretching
        target_pitch : np.array(shape=(frames,))
            The target pitch contour
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.

    Returns
        audio : np.array(shape=(samples,))
            The vocoded audio
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Time-stretch
        if constant_stretch is not None or \
           (isinstance(source_alignment, pypar.Alignment) and \
            isinstance(target_alignment, pypar.Alignment)):
            audio = time_stretch(audio,
                                 source_alignment,
                                 target_alignment,
                                 constant_stretch,
                                 fmin,
                                 fmax,
                                 sample_rate,
                                 tmpdir)

        # Pitch-shift
        if target_pitch is not None:
            audio = pitch_shift(
                audio, target_pitch, fmin, fmax, sample_rate, tmpdir)

        return audio


###############################################################################
# Utilities
###############################################################################


def get_manipulation(audio, fmin, fmax, sample_rate, tmpdir):
    """Retrieve a praat manipulation context

    Arguments
        audio : np.array(shape=(samples,))
            The speech signal to process
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
        sample_rate : int
            The audio sampling rate
        tmpdir : string
            Directory to save intermediate values

    Returns
        manipulation : parselmouth.Data
            The praat manipulation context
    """
    # Write audio to disk
    audio_file = os.path.join(tmpdir, 'audio.wav')
    save(audio_file, audio, sample_rate)

    # Setup the source utterance for modulation
    return praat.call(
        Sound(audio_file), "To Manipulation", 1e-3, fmin, fmax)


def pitch_shift(audio, pitch, fmin, fmax, sample_rate, tmpdir):
    """Perform praat pitch shifting on the manipulation

    Arguments
        audio : np.array(shape=(samples,))
            The speech signal to process
        pitch : np.array(shape=(frames,))
            The target pitch contour
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
        sample_rate : int
            The audio sampling rate
        tmpdir : string
            Directory to save intermediate values

    Returns
        audio : np.array(shape=(samples,))
            The pitch-shifted audio
    """
    # Don't edit in-place
    pitch = np.copy(pitch)

    # Convert unvoiced tokens to 0.
    pitch[np.isnan(pitch)] = 0.

    # Write pitch to disk
    pitch_file = os.path.join(tmpdir, 'pitch.txt')
    write_pitch_tier(pitch_file, pitch, float(len(audio)) / sample_rate)

    # Read pitch file into praat
    pitch_tier = Data.read(pitch_file)

    # Open a praat manipulation context
    manipulation = get_manipulation(audio, fmin, fmax, sample_rate, tmpdir)

    # Replace source pitch with new pitch
    praat.call([pitch_tier, manipulation], "Replace pitch tier")

    # Resynthesize
    return praat.call(manipulation, "Get resynthesis (overlap-add)").values[0]


def save(file, audio, sample_rate):
    """Save audio to disk"""
    soundfile.write(file, audio, sample_rate)


def time_stretch(audio,
                 alignment,
                 target_alignment,
                 constant_stretch,
                 fmin,
                 fmax,
                 sample_rate,
                 tmpdir):
    """Perform praat time stretching on the manipulation

    Arguments
        audio : np.array(shape=(samples,))
            The speech signal to process
        alignment : pypar.Alignment
            The current alignment if performing time-stretching
        target_alignment : pypar.Alignment
            The target alignment if performing time-stretching
        constant_stretch : float or None
            A constant value for time-stretching
        fmin : int
            The minimum allowable frequency in Hz.
        fmax : int
            The maximum allowable frequency in Hz.
        sample_rate : int
            The audio sampling rate
        tmpdir : string
            Directory to save intermediate values

    Returns
        audio : np.array(shape=(samples,))
            The time-stretched audio
    """
    if constant_stretch is not None:
        # Audio start and end time
        times = np.array([0., len(audio) / sample_rate])

        # Constant speed
        rates = np.array([1. / constant_stretch])

    else:
        # Phoneme start and end times
        times = np.array(
            [phoneme.start() for phoneme in alignment.phonemes()] +
            [alignment.end()])

        # Relative phoneme speeds
        rates = pypar.compare.per_phoneme_rate(alignment, target_alignment)
        rates = np.array(rates)
        rates[rates < .0625] = .0625

    # Write duration to disk
    duration_file = os.path.join(tmpdir, 'duration.txt')
    write_duration_tier(duration_file, times, rates)

    # Read duration file into praat
    duration_tier = Data.read(duration_file)

    # Open a praat manipulation context
    manipulation = get_manipulation(audio, fmin, fmax, sample_rate, tmpdir)

    # Replace phoneme durations
    praat.call([duration_tier, manipulation], 'Replace duration tier')

    # Resynthesize
    return praat.call(manipulation, "Get resynthesis (overlap-add)").values[0]


def write_duration_tier(filename, times, rates, eps=1e-6):
    """Write a duration tier to disk that is readable by praat

    Arguments
        filename : string
            Where to write the duration file
        times : np.array(shape=(phonemes + 1,))
            The original start and end times of each phoneme
        rates : np.array(shape=(phonemes,))
            The relative speed of the phoneme
        duration : float
            The duration of the audio in seconds
        eps : float
            Distance in seconds between two control points at a discontinuity
    """
    with open(filename, 'w') as file:
        # Write the header
        file.write(
            'File type = "ooTextFile"\nObject class = "DurationTier"\n\n')

        # Write the start and end of the audio file
        file.write(
            f'xmin = 0.000000\nxmax = {times[-1]:.6f}\npoints: size = {2 * len(times)}\n')

        # Start at the original rate
        file.write('points [1]:\n\tnumber = 0\n\tvalue = 1.000000\n')

        # Write the new duration
        for i, (start, end, rate) in enumerate(zip(times[:-1], times[1:], rates)):

            # We need 2 points to create a discontinuity in the automation curve
            file.write(f'points [{2 * i + 2}]:\n' +
                       f'\tnumber = {start + eps:.6f}\n' +
                       f'\tvalue = {rate:.6f}\n')
            file.write(f'points [{2 * i + 3}]:\n' +
                       f'\tnumber = {end - eps:.6f}\n' +
                       f'\tvalue = {rate:.6f}\n')

        # End at the original rate
        file.write(f'points [{2 * len(times)}]:\n' +
                   f'\tnumber = {times[-1]:.6f}\n' +
                   '\tvalue = 1.000000\n')


def write_pitch_tier(filename, pitch, duration):
    """
    Write a pitch tier to disk that is readable by praat

    Arguments
        filename : string
            Where to write the pitch file
        pitch : np.array
            The new pitch contour to use for synthesis
        duration : float
            The duration of the audio in seconds
    """
    times = np.linspace(0., duration, len(pitch))
    with open(filename, 'w') as file:
        # Write the header
        file.write('File type = "ooTextFile"\nObject class = "PitchTier"\n\n')

        # Write the start and end of the audio file
        file.write('0\n')
        file.write(str(duration) + '\n')

        # Write the number of voiced frames
        file.write(str(np.sum(~np.isnan(pitch))) + '\n')

        # Write the pitch values and time points
        for time, value in zip(times, pitch):
            if not np.isnan(value):
                file.write(str(time) + '\n' + str(value) + '\n')
