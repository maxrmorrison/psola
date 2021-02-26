import argparse

import psola


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # File locations
    parser.add_argument('--audio_files',
                        nargs='+',
                        help='The speech signal to process')
    parser.add_argument('--source_alignment_files',
                        nargs='+',
                        help='The files containing the original alignments')
    parser.add_argument('--target_alignment_files',
                        nargs='+',
                        help='The files containing the target alignments')
    parser.add_argument('--constant_stretch',
                        type=float,
                        help='A constant value for time-stretching')
    parser.add_argument('--target_pitch_files',
                        nargs='+',
                        help='The target pitch contour')

    # DSP parameters
    parser.add_argument('--fmin',
                        type=int,
                        default=40,
                        help='The minimum allowable frequency in Hz')
    parser.add_argument('--fmax',
                        type=int,
                        default=500,
                        help='The maximum allowable frequency in Hz')

    # Output file location
    parser.add_argument('--output_files',
                        nargs='+',
                        help='Where to save the vocoded audio')

    return parser.parse_args()


if __name__ == '__main__':
    psola.from_files_to_files(**vars(parse_args()))
