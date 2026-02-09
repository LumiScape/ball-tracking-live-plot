import argparse
import logging
from collections import deque
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from ball_tracking.colormap import colormap_rainbow
from ball_tracking.core import Point2D
from ball_tracking.video_loop import VideoLoop
from ball_tracking.thread_vid_writter import ThreadedVideoWriter
import time

def parse_video_source(value: str) -> Path | int:
    # If the input is just digits, return it as an int
    if value.isdigit():
        return int(value)
    # Otherwise, return it as a Path
    return Path(value)

def write_video(video_path: Path|int, video_loop) -> cv2.VideoWriter:
    video_writer = None

    if isinstance(video_path, Path):
        filename = str(video_path.with_name(video_path.stem + "_tracked.mp4"))
    else:
        filename = "camera_output_tracked.mp4"

    return ThreadedVideoWriter(
        filename=filename,
        fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
        fps=video_loop.fps,
        frame_size=video_loop.video_resolution,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=parse_video_source,
        default=Path("media/ball3.mp4"),
        help="Path to the video or camera index",
    )
    parser.add_argument(
        "--alpha-blending",
        action="store_true",
        default=False,
        help="Use alpha blending to smooth the trajectory",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=40,
        help="Number of frames to keep in the trajectory",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=4.5,
        help="Number of seconds to skip in the video",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the video",
    )
    parser.add_argument(
        "--show-masks",
        action="store_true",
        help="Show the masks used for filtering",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the video with the tracked ball",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=120,
        help="Requested FPS for camera (default: 120)",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Enable stereo camera mode (e.g., for ELP 3D Stereo USB Camera)",
    )
    parser.add_argument(
        "--stereo-right",
        type=int,
        default=None,
        help="Camera index for right stereo camera (e.g., if left is 0, right is typically 1)",
    )
    parser.add_argument(
        "--stereo-use",
        type=str,
        choices=["left", "right", "both"],
        default="left",
        help="Which camera(s) to use in stereo mode: 'left', 'right', or 'both' (side-by-side)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    video_path = args.video_path

    scale = 0.5
    with VideoLoop(
        video_path,
        loop=args.loop,
        skip_seconds=args.skip_seconds,
        camera_fps=args.camera_fps,
        stereo_mode=args.stereo,
        stereo_camera_right=args.stereo_right,
        stereo_use=args.stereo_use,
    ) as video_loop:
        logger.info(f"Loaded video: {video_path}, resolution: {video_loop.video_resolution}, fps: {video_loop.fps}")

        if args.save_video :
            video_writer = write_video(video_path=video_path, video_loop=video_loop)
        else:
            video_writer = None

        # initialize background model
        bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=128, detectShadows=False)

        tracked_pos: deque[Point2D] = deque(maxlen=args.trajectory_length)

        video_loop.reset()
        _, frame0 = next(video_loop)
        bg_sub.apply(frame0, learningRate=1.0)
        
        fps_frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0

        for wait_time, frame in video_loop:
            if frame is None: continue
            loop_start = time.perf_counter()

            frame_annotated = frame.copy() # 1920 x 1080

            # --- STEP 1: Downscale for logic ---
            frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)


            # filter based on color
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 60, 60])
            upper_yellow = np.array([50, 255, 255])

            mask_color = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_color = cv2.morphologyEx(
                mask_color,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )

            # filter based on motion
            mask_fg = bg_sub.apply(frame_small, learningRate=0)
            mask_fg = cv2.dilate(
                mask_fg,
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )

            # combine both masks
            mask = cv2.bitwise_and(mask_color, mask_fg)
            mask = cv2.morphologyEx(
                mask,
                op=cv2.MORPH_OPEN,
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )

            # find largest contour corresponding to the ball we want to track
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_small = (x + w // 2, y + h // 2)
                center = (int(center_small[0] / scale), int(center_small[1] / scale))

                if len(tracked_pos) > 0:
                    # smooth the trajectory
                    prev_center = tracked_pos[-1]
                    alpha = 0.9
                    center = (
                        int((1 - alpha) * prev_center[0] + alpha * center[0]),
                        int((1 - alpha) * prev_center[1] + alpha * center[1]),
                    )

                tracked_pos.append(center)

                cv2.circle(frame_annotated, center, 30, (255, 0, 0), 2)
                cv2.circle(frame_annotated, center, 2, (255, 0, 0), 2)


            # --- Update FPS Logic ---
            fps_frame_count += 1
            
            # Every 30 frames, recalculate the "Instant" FPS
            if fps_frame_count >= 30:
                end_time = time.time()
                current_fps = fps_frame_count / (end_time - fps_start_time)
                
                # Print stats
                loop_end = time.perf_counter()
                processing_ms = (loop_end - loop_start) * 1000
                print(f"Tracking: {processing_ms:.2f}ms | Instant FPS: {current_fps:.1f}")
                
                # Reset window
                fps_frame_count = 0
                fps_start_time = time.time()

            # draw trajectory
            traj_len = len(tracked_pos)
            for i, (p1, p2) in enumerate(pairwise(tracked_pos)):
                norm_idx = i / traj_len
                color = colormap_rainbow(norm_idx)

                if args.alpha_blending:
                    # Create a temporary image to draw the line
                    temp = frame_annotated.copy()
                    cv2.line(temp, pt1=p1, pt2=p2, color=color, thickness=2)

                    # Blend the temporary image with the original frame
                    cv2.addWeighted(
                        temp,
                        norm_idx,
                        frame_annotated,
                        1 - norm_idx,
                        0,
                        frame_annotated,
                    )
                else:
                    cv2.line(frame_annotated, pt1=p1, pt2=p2, color=color, thickness=2)

            if video_writer is not None:
                video_writer.write(frame_annotated)
            preview = cv2.resize(frame_annotated, (1280, 720))
            cv2.imshow("120 fps with 1920 x 1080", preview)

            if args.show_masks:
                cv2.imshow("Mask FG", mask_fg)
                cv2.imshow("Mask Color", mask_color)

            actual_wait = 1 if isinstance(video_path, int) else wait_time
            key = cv2.waitKey(actual_wait) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                video_loop.reset()

        if video_writer is not None:
            video_writer.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
