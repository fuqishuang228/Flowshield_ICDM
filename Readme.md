# FlowShield: Open Science Release

This repository provides the **code** for our paper:  
**Follow the Flow: How FlowShield Sniffs Out Dirty Money on Blockchain**

---

## Configuration

Set the following environment variables before running scripts that access external services:

- `INFURA_PROJECT_ID`
- `ETHERSCAN_API_KEY`
- `OPENAI_API_KEY`

Never commit API keys or local `.env` files.

## 📂 Code

The `Code/` directory contains four stage-specific folders and the final model:

- **`P1_*` – `P4_*`**  
  - Correspond to the first three stages of **FlowShield**.  
  - Each folder implements the corresponding processing and training step.

- **`P4_SARs_generation/S1_FlowShield.py`**  
  - The **main detection script** for identifying money laundering activities.  
  - 👉 If you want a quick start, simply run this script, as we have already provided the necessary input data.

---

## 🚀 Usage

### Quick Start
```bash
python Code/P4_SARs_generation/S1_FlowShield.py
