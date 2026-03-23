import logging


class LoggerFactory:
    def __init__(
        self,
        level=logging.INFO,
        log_format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    ):
        self.level = level
        self.log_format = log_format

    def get_logger(self, name=__name__):
        logging.basicConfig(level=self.level, format=self.log_format)
        return logging.getLogger(name)


_logger_factory = LoggerFactory()


def get_logger(name=__name__):
    return _logger_factory.get_logger(name)
