# QERA: an Analytical Framework for Quantization Error Reconstruction

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.06040-b31b1b.svg)](https://arxiv.org/abs/2410.06040)
[![ic](https://img.shields.io/badge/Imperial-DeepWokLab-0D80D8)
](https://deepwok.github.io/)
[![license](https://img.shields.io/badge/License-Apache%202.0-D22128.svg)](/LICENSE)

</h5>


<h5 align="center">
<img src="./docs/logo.png" width="200">
</h5>

This is the official implementation of the ICLR'25 paper "QERA: an Analytical Framework for Quantization Error Reconstruction".
In this paper, we solve the following problem:

Given a trained layer $\mathbf{y}=\mathbf{x}W$, QERA derives the closed-form solution to the matrix $C_k$ in $\tilde{\mathbf{y}}=\mathbf{x}(W_q + C_k)$, where $W_q$ is the approximated weight matrix (high-rank but low-precision/sparse/...) and $C_k$ is low-rank but high-precision., to minimize the expectation of the layer output error $E[\Vert \tilde{\mathbf{y}} - \mathbf{y} \Vert_2^2]$.

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

## Experiment Scripts

See [`experiments/ptq`](/experiments/ptq/) and [`experiments/qpeft`](/experiments/qpeft/) for PTQ and qLoRA fine-tuning experiments, respectively.

## Citation

```bibtex
@article{zhang2024qera,
  title={QERA: an Analytical Framework for Quantization Error Reconstruction},
  author={Zhang, Cheng and Wong, Jeffrey TH and Xiao, Can and Constantinides, George A and Zhao, Yiren},
  journal={arXiv preprint arXiv:2410.06040},
  year={2024}
}
```