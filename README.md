# SQL_LoRA+QLoRA Project Documentation
## Overview
SQL_LoRA+QLoRA is a project that implements Low-Rank Adaptation (LoRA) and Quantized Low-Rank Adaptation (QLoRA) techniques for SQL-related machine learning tasks. This document provides an overview of the project structure and components.

## Project Structure

- SQL_LoRA+QLoRA/
  - project.json              — Project configuration file
  - assets/                   — Project assets directory
    - .METADATA/              — Metadata for data assets
    - data_asset/             — Data assets for training and evaluation
    - notebook/               — Jupyter notebooks for demonstrations
  - assettypes/               — Definitions for asset types
    - model_entry_user.json   — User model entry configuration
    - modelfacts_user.json    — User model facts configuration
    - wx_prompt.json          — Prompt templates configuration


## Components
project.json
The main configuration file that defines project parameters, dependencies, and settings.

### Assets Directory
.METADATA: Contains metadata JSON files for each data asset used in the project
data_asset: Contains the actual data files used for model training and evaluation
notebook: Jupyter notebooks demonstrating model usage, training procedures, and evaluation results
### Asset Types
model_entry_user.json: Defines the schema for user model entries
modelfacts_user.json: Defines the structure for capturing model facts and metrics
wx_prompt.json: Defines prompt templates used for model interactions

## About LoRA and QLoRA
LoRA (Low-Rank Adaptation) and QLoRA (Quantized Low-Rank Adaptation) are fine-tuning techniques that enable efficient adaptation of large language models with significantly fewer parameters than full fine-tuning.

LoRA: Updates low-rank decomposition matrices instead of the full model weights
QLoRA: Combines quantization with LoRA to further reduce memory requirements
This project applies these techniques specifically to SQL-related tasks.
