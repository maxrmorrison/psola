# Time-domain pitch-synchronous overlap-add (TD-PSOLA)
[![PyPI](https://img.shields.io/pypi/v/psola.svg)](https://pypi.python.org/pypi/psola)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- [![Downloads](https://pepy.tech/badge/psola)](https://pepy.tech/project/psola) -->

This module permits contant- and variable-rate pitch-shifting and
time-stretching of speech. It is a wrapper around the `parselmouth` [1]
wrapper around the Praat [2] implementation of TD-PSOLA [3]. Pitch-shifting
is performed by providing a numpy array of target pitch values equally spaced
over time. Variable-rate time stretching uses forced phoneme alignment via
[`pypar`](https://github.com/maxrmorrison/pypar).

If you need to extract pitch features or phoneme alignments, see
[`torchcrepe`](https://github.com/maxrmorrison/torchcrepe) for pitch estimation
and [`pyfoal`](https://github.com/maxrmorrison/pyfoal) for forced alignment.
If you only want to perform pitch-shifting, you do not need to extract
forced alignments. If you want to do variable-rate time stretching, you do not
need to perform pitch estimation.


## Installation

`pip install psola`


## Usage

If you want to perform pitch-shifting or time-stretching on audio already
loaded into memory, use `psola.vocode`. If you want to do this with audio
saved in a file, use `psola.from_file`. You can use `psola.to_file` or
`psola.from_file_to_file` to save the results to a file. To process many
files at once with multiprocessing, use `psola.from_files_to_files`.
Each of these functions is documented below. The command-line interface
wraps the arguments of `psola.from_files_to_files` and is described in
the next section.


### `psola.vocode`

```python
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
```


### `psola.from_file`

```python
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
```


### `psola.to_file`

```python
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
```


### `psola.from_file_to_file`

```python
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
```


### `psola.from_files_to_files`

```python
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
```


## Command-line interface

```
usage: python -m psola
    [-h]
    [--audio_files AUDIO_FILES [AUDIO_FILES ...]]
    [--source_alignment_files SOURCE_ALIGNMENT_FILES [SOURCE_ALIGNMENT_FILES ...]]
    [--target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]]
    [--constant_stretch CONSTANT_STRETCH]
    [--target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]]
    [--fmin FMIN]
    [--fmax FMAX]
    [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The speech signal to process
  --source_alignment_files SOURCE_ALIGNMENT_FILES [SOURCE_ALIGNMENT_FILES ...]
                        The files containing the original alignments
  --target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]
                        The files containing the target alignments
  --constant_stretch CONSTANT_STRETCH
                        A constant value for time-stretching
  --target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]
                        The target pitch contour
  --fmin FMIN           The minimum allowable frequency in Hz
  --fmax FMAX           The maximum allowable frequency in Hz
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        Where to save the vocoded audio
```


## References

[1] Y. Jadoul, B. Thompson, and B. De Boer, "Introducing parselmouth: A python interface to praat," Journal of Phonetics, vol. 71, pp. 1–15, 2018.

[2] P. Boersma, "Praat: doing phonetics by computer", http://www.praat.org/, 2006.

[3] E. Moulines and F. Charpentier, "Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones," Speech communication, 1990.
