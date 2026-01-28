import queue
import cv2
import threading


class ThreadedVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=128) # Buffer up to 128 frames
        self.stopped = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1.0)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def write(self, frame):
        if not self.stopped:
            try:
                self.queue.put_nowait(frame) # Don't wait if queue is full
            except queue.Full:
                pass # Drop frame if disk/CPU can't keep up

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.writer.release()