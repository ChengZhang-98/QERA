import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[0].joinpath("src").as_posix())


from qera.ptq_pipeline import (
    pipeline_qera,
    pipeline_fp16_bf16_fp32,
    pipeline_q_baseline,
    chunk_checker,
)
from qera.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    chunk_checker()
