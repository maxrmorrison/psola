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
    parser.add_argument('--target_alignment_files',
                        nargs='+',
                        help='The target phoneme alignment')
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
    parser.add_argument('--hopsize',
                        type=int,
                        default=psola.HOPSIZE,
                        help='The hopsize of the input pitch in milliseconds')

    # Intermediary file locations
    parser.add_argument('--tmpdir',
                        help='Directory to save intermediate values. ' +
                             'Defaults to the system default.')

    # Output file location
    parser.add_argument('--output_files',
                        nargs='+',
                        help='Where to save the vocoded audio')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Vocode and save to disk
    psola.from_files_to_files(args.audio_files,
                              args.output_files,
                              args.target_alignment_files,
                              args.target_pitch_files,
                              args.fmin,
                              args.fmax,
                              args.hopsize,
                              args.tmpdir)
