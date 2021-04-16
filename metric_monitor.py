
class MetricMonitor:
    def __init__(self, metric_name, float_precision):
        self.metric_name = metric_name
        self.float_precision = float_precision
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.metric_name}: {self.avg:.{self.float_precision}f}"

