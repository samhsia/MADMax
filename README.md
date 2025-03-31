# MADMax

This repository is for open-sourcing and artifact evaluation of the International Symposium on Computer Architecture (ISCA) 2024 paper **MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems**.

## Setup

Run `./setup.sh`

## Functional Tests

For a DLRM example, run `python run_model.py`.

For an LLM example, run `python run_model.py --model-cfg-file 'model_cfgs/llm/llama2_70b.json' --system-cfg-file 'system_cfgs/dc_a/dc_a_2048.json' --task-cfg-file 'task_cfgs/llm/llm_train.json'`

For each example, make sure that you are able to see printed outputs that end with `**************************************************`.

## Artifact Evaluation

The two main folders for artifact evaluation are:

- `artfiact_notebooks` - Jupyter Notebooks used for launching experiments to recreate performance model results
  - `[0] Cloud Provider Launcher.ipynb` - paper Figures 1 and 16
  - `[1] DLRM A Validation.ipynb` - paper Figure 7

- `artifact_sheets` - Microsoft Excel sheets that contain organized data of experiment results
  - `[A] Cloud Provider Results.xlsx` - paper Figures 1 and 16

## Codebase Structure

