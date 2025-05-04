import time
import threading

class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if not self._running:
                self._start_time = time.time()
                self._running = True
                self._paused = False

    def stop(self):
        with self._lock:
            if self._running:
                if not self._paused:
                    self._elapsed_time += time.time() - self._start_time
                self._running = False
                self._paused = False
                self._elapsed_time = 0

    def pause(self):
        with self._lock:
            if self._running and not self._paused:
                self._elapsed_time += time.time() - self._start_time
                self._paused = True

    def resume(self):
        with self._lock:
            if self._running and self._paused:
                self._start_time = time.time()
                self._paused = False
            

    def get_elapsed_time(self):
        with self._lock:
            if self._running and not self._paused:
                return self._elapsed_time + (time.time() - self._start_time)
            return self._elapsed_time
        