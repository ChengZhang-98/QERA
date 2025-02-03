# QERA: Analytical Solution to Quantization Error Approximation


## Env Setup

```bash
conda env create -f environment.yml
conda run -n qera python -m pip install -r requirements.txt
```

## Usage


1. MXINT4 weight quantization only, no qera

    ```bash
    python ptq_pipeline.py ./experiments/configs/w-only-uniform-rank.yaml --disable-qera --disable-perplexity-eval
    ```

    The config template `w-only-uniform-rank.yaml` runs TinyLlama on a subset of SlimPajama for calibration, and WikiText2 for perplexity evaluation.

2. MXINT4 weight, scale = identity matrix (ZeroQuantV2)

    ```bash
    python ptq_pipeline.py ./experiments/configs/w-s-uniform-rank.yaml --qera-scaling-mode identity --disable-perplexity-eval
    ```

    "scale = idenity matrix" means that we just apply SVD to the quantization error: $\mathrm{SVD}(W - W_q)$.

3. MXINT weight, scale = activation induced diagonal matrix, which is derived by assuming $E[x_i x_j] = 0$ for $i\neq j$.

    ```bash
    python ptq_pipeline.py ./experiments/configs/w-s-activation-rank.yaml --qera-scaling-mode diag --disable-perplexity-eval
    ```

4. MXINT weight, scale = auto-correlation matrix of activation vectors, which is derived without the assumption.

    ```bash
    python ptq_pipeline.py ./experiments/configs/w-s-activation-rank.yaml --qera-scaling-mode rxx --disable-perplexity-eval
    ```


