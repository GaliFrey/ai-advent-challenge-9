"""One process-wide admission, no queue; retain it while abandoned CPU work runs."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

_ADMISSION = Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='support-retrieval')


class Lease:
    def __init__(self):
        self._mutex = Lock()
        self._worker = None
        self._closed = False
        self._released = False

    def submit(self, function, *args):
        with self._mutex:
            self._worker = _EXECUTOR.submit(function, *args)
        self._worker.add_done_callback(lambda _: self._release_if_ready())
        return self._worker

    def _release_if_ready(self):
        with self._mutex:
            if self._closed and not self._released and (self._worker is None or self._worker.done()):
                self._released = True
                _ADMISSION.release()

    def close(self):
        with self._mutex:
            self._closed = True
        self._release_if_ready()


def admit():
    return Lease() if _ADMISSION.acquire(blocking=False) else None
