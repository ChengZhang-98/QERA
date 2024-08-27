import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[0].joinpath("src").as_posix())


from loqer.peft_pipeline_v2 import adapt_and_save_pipeline
from loqer.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    adapt_and_save_pipeline()
