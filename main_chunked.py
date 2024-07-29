import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[0].joinpath("src").as_posix())


from loqer.pipeline import pipeline_loqer,pipeline_loqer_chunked
from loqer.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    pipeline_loqer_chunked()
