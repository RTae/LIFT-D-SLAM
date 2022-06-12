import argparse

def make_parser():

    parser = argparse.ArgumentParser("LIFT-D-SLAM sample")
    parser.add_argument(
        "-v",
        "--video_path",
        required=True,
        type=str,
        
        help="Input your video file to test.",
    )

    return parser