import builtins


class StaticPrinter:
    def __init__(self):
        self.num_lines = 0

    def print(self, *line):
        print(*line)
        self.num_lines += 1

    def reset(self):
        for _ in range(self.num_lines):
            print("\033[F", end="")  # Cursor up one line
            print("\033[K", end="")  # Clear to the end of line
        self.num_lines = 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """

    builtin_print = builtins.print

    def print_new(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print_new


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg
