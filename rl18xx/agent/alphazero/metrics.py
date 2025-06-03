from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from threading import Lock
import logging

LOGGER = logging.getLogger(__name__)

class Metrics:
    def __init__(self, log_dir):
        self.summary_writer = SummaryWriter(log_dir)
        self.metric_step = defaultdict(int)
        self.lock = Lock()

    def close(self):
        self.summary_writer.close()

    def add_scalar(self, name, value, loop, game=None):
        with self.lock:
            metric_step = self.metric_step[name]
            self.summary_writer.add_scalar(name, value, metric_step)
            self.metric_step[name] += 1

    def add_histogram(self, name, value, loop, game=None):
        with self.lock:
            metric_step = self.metric_step[name]
            self.summary_writer.add_histogram(name, value, metric_step)
            self.metric_step[name] += 1
