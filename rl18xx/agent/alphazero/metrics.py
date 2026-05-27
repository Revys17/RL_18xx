from torch.utils.tensorboard import SummaryWriter
from threading import Lock
import logging

LOGGER = logging.getLogger(__name__)

class Metrics:
    def __init__(self, log_dir):
        self.summary_writer = SummaryWriter(log_dir)
        self.lock = Lock()

    def close(self):
        self.summary_writer.close()

    def add_scalar(self, name, value, loop, game=None):
        """Write a scalar to TensorBoard.

        The caller is responsible for supplying the appropriate global step.
        Typically pass ``loop`` for per-iteration metrics. For finer-grained
        series, pass ``game`` (per self-play game) or some other move-level
        counter as ``game``; when provided it takes precedence as the step.
        """
        step = game if game is not None else loop
        with self.lock:
            self.summary_writer.add_scalar(name, value, step)

    def add_histogram(self, name, value, loop, game=None):
        """Write a histogram to TensorBoard.

        Same step semantics as ``add_scalar``: pass ``loop`` for per-iteration
        histograms, ``game`` for per-game (or finer) granularity.
        """
        step = game if game is not None else loop
        with self.lock:
            self.summary_writer.add_histogram(name, value, step)
