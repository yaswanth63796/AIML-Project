# preprocess.py
import argparse
from utils import prepare_image_folders_from_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="Data", help="Source video folder (Adult/ Child/)")
    parser.add_argument("--dest", type=str, default="Data_frames", help="Destination for frames with train/val")
    parser.add_argument("--frame_step", type=int, default=10, help="Save every Nth frame from each video")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per video (optional)")
    args = parser.parse_args()

    prepare_image_folders_from_videos(args.src, args.dest, frame_step=args.frame_step, max_frames_per_video=args.max_frames)
