# QERA: an Analytical Framework for Quantization Error Reconstruction

[[paper]](https://arxiv.org/abs/2410.06040)

🍍 This is the official implementation of the ICLR'25 paper "QERA: an Analytical Framework for Quantization Error Reconstruction".

![cover](/docs/qera-cover.png)

## Env Setup

```bash
git clone git@github.com:ChengZhang-98/QERA.git
cd QERA
git submodule update --init
conda env create -f environment.yml
conda activate qera
pip install -r requirements.txt
pip install -e .
```

## Entry Points

In the source code and scripts, we use the following abbreviations for the low-rank term types:
- If `--disable-qera` is set, no low-rank terms are used, i.e., weight-only quantization.
- Else:
    - `identity`: Truncated SVD on the quantized weight matrix, i.e., [ZeroQuant-V2](https://arxiv.org/abs/2303.08302)
    - `lqer`: The heuristic method proposed in [LQER paper](https://arxiv.org/abs/2402.02446)
    - `diag`: QERA-approx in our paper.
    - `exact`: QERA-exact in our paper.


### Post-Training Quantization

- `ptq_bf16_baseline.py` evaluates BF16 baseline.
- `ptq_q_baseline.py` evaluates PTQ baseline.
- `ptq_pipeline.py` runs data calibration (if needed), computes low-rank terms, and evaluates the quantized model.
- `ptq_pipeline_chunked.py` runs data calibration (if needed), and computes low-rank terms for a chunk of layers. This is useful for large models. If all chunks (layers) are computed, this script also triggers the evaluation of the quantized model.
    - `chunk_checker.py` checks the completion of the chunks (optional).

### Quantized LoRA Fine-Tuning

- `adapt_and_save.py` run data calibration, quantizes the model, computes the initial value of the low-rank terms, and saves the quantized model + low-rank terms.
- `glue_train.py` fine-tunes the qLoRA-adapted model with low-rank terms on GLUE tasks.
- `clm_train.py` fine-tunes the qLoRA-adapted model with low-rank terms on WikiText2.
- `gsm8k_train.py` fine-tunes the qLoRA-adapted model with low-rank terms on GSM8K.

### Experiment Scripts

See [`experiments/ptq`](/experiments/ptq/) and [`experiments/qpeft`](/experiments/qpeft/) for PTQ and qLoRA fine-tuning experiments, respectively.