### LLM4Causal Set Up Guide

STEP 1: install dependencies - please fun following bash commands in a terminal

```
pip install -r requirements.txt
```

STEP 2: login huggingface-hub to access LLama 2 - please run the following in bash with your KEY

```
huggingface-cli login ### need to request Llama 2 from HF
```

STEP 3: setting accelerate - still in bash, run

```
accelerate config
```

### LLM4Causal Running Guide

STEP 1-3: install dependencies as in STEP 1-3 in `Set Up Guide`

STEP 4: trigger scrip and run the model - note that you may change the args by passing "--model_path" etc.
```
accelerate launch llm4causal_e2e.py
```

### LLM4Causal Experiment and Data Regeneration Guide

STEP 1-3: install dependencies as in STEP 1-3 in `Set Up Guide`

STEP 4: follow step 0 - 10 accordingly, critical data files from our run is under data/ folder.


### Folder Descriptions

- **src/**: Contains the LLM4Causal source code for end-to-end execution and tuning data generation.
- **model/**: Includes the three LLM4Causal model checkpoints used in the paper.
- **data_work/**: Demo CSV files for running end-to-end examples.
- **data/**: Contains two benchmark datasets and key intermediate files.
- **llm4causal_e2e.py**: A script to run the full LLM4Causal end-to-end pipeline.
