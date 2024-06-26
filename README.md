# Closed-Form Solution to Activation-Informed Weight Quantization Error Approximation

![](/docs/big-little-llama-small.png)


## Env Setup

```bash
conda env create -f environment.yml
conda run -n loqer python -m pip install -r requirements.txt
```

## Usage

Add `LoQER/src` to PYTHONPATH.

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
```

1. MXINT4 weight quantization only, no loqer

    ```bash
    python main.py ./experiments/configs/w-only-uniform-rank.yaml --disable-loqer --disable-lm-eval
    ```

    The config template `w-only-uniform-rank.yaml` runs TinyLlama on a subset of SlimPajama for calibration, and WikiText2 for perplexity evaluation.

2. MXINT4 weight, scale = identity matrix

    ```bash
    python main.py ./experiments/configs/w-s-uniform-rank.yaml --loqer-scaling-mode identity --disable-lm-eval
    ```

    "scale = idenity matrix" means that we just apply SVD to the quantization error: $\mathrm{SVD}(W - W_q)$.

3. MXINT weight, scale = activation induced diagonal matrix, which is derived by assuming $E[x_i x_j] = 0$ for $i\neq j$.

    ```bash
    python main.py ./experiments/configs/w-s-activation-rank.yaml --loqer-scaling-mode diag --disable-lm-eval
    ```

4. MXINT weight, scale = auto-correlation matrix of activation vectors, which is derived without the assumption.

    ```bash
    python main.py ./experiments/configs/w-s-activation-rank.yaml --loqer-scaling-mode rxx --disable-lm-eval
    ```


## Reference

- [LQER: Low-Rank Quantization Error Reconstruction for LLMs](https://arxiv.org/abs/2402.02446)
- [Closed-form solution to the scale matrix](https://typst.app/project/rQcqVZNgJGJz2LLuOrZx6y). ⚠️ **Not yet published. Please do not distribute** ⚠️

