# Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search

This repository contains implementation for the paper "[Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search](https://arxiv.org/abs/2508.12211)".


## Installation

Clone the repository:

```bash
git clone https://github.com/cyrusneary/vlaps
cd vlaps
```

This repository requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

Install UV, a package and project manager: https://docs.astral.sh/uv/getting-started/installation/

Create and activate a new virtual environment:

```bash
cd VLAPS
uv venv .venv --python 3.10
source .venv/bin/activate
```

Install Octo:

```bash
cd ../third_party/octo
uv pip install -e .
uv pip install -r requirements.txt
uv pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then install LIBERO:
```bash
cd ../LIBERO
uv pip install cmake==3.24.3
uv pip install -r requirements.txt
uv pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install -e .
```

Install VLAPS requirements and libraries.

```bash
cd ../../VLAPS
uv pip install -r requirements.txt
uv pip install -e .
```

Add libero to the python path
```bash
cd ..
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO
```

## Usage

### Running experiments

```bash
uv run VLAPS/run_libero_experiment.py --config-name=config_vlaps_octo.yaml
```

### Configuring experiments

This repository uses [Hydra](https://hydra.cc/) to configure expeirments. To edit the configurations, edit the yaml files in the VLAPS/config directory.

Before running the first experiment, you will need to edit the Octo model checkpoint path in VLAPS/config/agent_config/octo_agent.yaml and in VLAPS/config/agent_config/vlaps_octo_agent_config.yaml.

### Finetuning Octo

To Finetune octo, navigate to the octo repository and run the finetuning script.

```bash
cd third_party/octo
uv run scripts/finetune_libero.py
```

Please refer to the README.md file in third_party/octo for more details.
