import time
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
import threading

class VideoLoop:
    cap: cv2.VideoCapture
    fps: int
    frame_count: int
    width: int
    height: int
    frame_time: float
    video_resolution: tuple[int, int]
    last_frame_time: float

    def __init__(
        self,
        video_path: str | Path | int,
        loop: bool = False,
        skip_seconds: float = 0.0,
    ) -> None:
        self.video_path = video_path
        self.loop = loop
        self.skip_seconds = skip_seconds

        # Threading
        self.stopped = False
        self.frame = None
        self.ret = False
        self.thread = None

    def _update(self) -> None:
        """Internal method to grab frames in seperate thread."""
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                self.stopped = True

    def __iter__(self) -> "VideoLoop":
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        """
        Read the next frame from the video and calculate the time to wait

        Returns
            tuple[int, np.ndarray]:
                - time to wait in milliseconds
                - frame read from the video
        """
        # Case 1 Camera
        if isinstance(self.video_path, int):
            if self.stopped:
                raise StopIteration
            # No wait for live camera
            return 1, self.frame if self.frame is not None else np.zeros((self.height, self.width, 3), np.uint8)
        
        # Case 2 Video
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.__next__()
            else:
                self.cap.release()
                raise StopIteration

        # calculate the time elapsed since the last frame
        current_frame_time = time.time()
        dt = (current_frame_time - self.last_frame_time) * 1000
        self.last_frame_time = current_frame_time

        sleep_time = max(1, int(self.frame_time - dt))

        return sleep_time, frame

    def __del__(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoLoop":
        # Check if video_path is int (Camera Index)
        if isinstance(self.video_path, int):
            # 1 Open Camera with DSHOW
            self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_DSHOW)

            # 2 Apply specific Camera settings
            # 0.25 usually means 'Manual Mode' on Windows
            #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            #self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)
            #self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            #self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # requested width # 1920 x 1080 or 1280 x 720
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # requested height
            self.cap.set(cv2.CAP_PROP_FPS, 120)          # requested FPS
        
        else:
            self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Error: cannot read video file {self.video_path}")

        # get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0: self.fps = 30 # Fallbackvalue
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # calculate utility variables
        self.frame_time = 1000 / self.fps
        self.last_frame_time = time.time()
        self.video_resolution = (self.width, self.height)
        if isinstance(self.video_path, int):
            self.stopped = False
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        else:
            if self.skip_seconds > self.frame_count / self.fps:
                raise ValueError(
                    f"Error: skip_seconds ({self.skip_seconds:.2f}s) is greater than the video duration ({self.frame_count / self.fps:.2f}s)"
                )
            self.reset()

        return self

    def __exit__(
        self,
        _: type[BaseException] | None,
        __: BaseException | None,
        ___: TracebackType | None,
    ) -> None:
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        self.cap.release()

    def reset(self) -> None:
        if not isinstance(self.video_path, int):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps * self.skip_seconds))
        self.last_frame_time = time.time()
