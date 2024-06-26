from loqer.pipeline import pipeline_loqer
from loqer.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    pipeline_loqer()
