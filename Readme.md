# FLOWSHIELD

This is the source code of the ICDM '26 paper **"FLOWSHIELD: Cryptocurrency anti-money laundering with transaction semantics parsing and fund flow tracking"**.

![Overview of the FLOWSHIELD framework](assets/framework.png)

## Configuration

Set the following environment variables before running scripts that access external services:

- `INFURA_PROJECT_ID`
- `ETHERSCAN_API_KEY`
- `OPENAI_API_KEY`

## Data

Due to repository storage limitations, the data are hosted separately. BybitML is publicly available on [OSF](https://osf.io/nx7aj/overview?view_only=663038fbea43491bb4010011c7f28b23).

## Code

The code for Bitcoin and Ethereum is provided in [`FlowShield_BTC/`](FlowShield_BTC/) and [`FlowShield_ETH/`](FlowShield_ETH/), respectively.

---

## Usage

### Quick Start

Download the data, prepare the required embeddings, and update the input and output paths in the scripts to match your local setup. Then run the main detection script:

```bash
python FlowShield_ETH/P4_SARs_generation/S1_FlowShield.py
```

## Citation

If you compare with, build on, or use aspects of this work, please cite the following:

```bibtex
@inproceedings{fu2026flowshield,
  title={{FLOWSHIELD}: Cryptocurrency anti-money laundering with transaction semantics parsing and fund flow tracking},
  author={Fu, Qishuang and Deppeler, Andreas and Liu, Joseph K. and Liu, Yixin and Pan, Shirui and Wang, Qin and Wang, Weiqing and Yuen, Tsz Hon},
  booktitle={ICDM},
  year={2026}
}
```
