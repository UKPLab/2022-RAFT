## Transformers with learnable activation functions

### Structure
```
    ├── args.py                   // Includes all arguments used in this project
    ├── callbacks.py              // wandb Callbacks
    ├── outputs                   // the directory to save model, you need to create it by yourself.
    ├── requirements.txt          // Library dependency
    ├── configs                   // model configuration for pre-training
    ├── run_glue.py               // entrypoint for GLUE benchmark
    ├── run_squad.py              // entrypoint for SQuAD
    ├── run_mlm.py                // entrypoint for pretraining
    ├── run_plot.py               // plot RAFs for different tasks and layers
    ├── schedules.py              // a collection of different LR schedulers
    ├── scripts                   // slurm scripts
    ├── triviaqa                  // convert triviaqa to squad format
    ├── customtrainer.py          // used for pre-training
    ├── transformers_modified     // modified transformers library
    
    
```

### Dependency
 ```python
pip install -r requirements.txt
```

- Rational Functions 
Please look into https://github.com/ml-research/rational_activations. If you want to install cuda acceleration version, set `export "FORCE_CUDA"=1`
- gcc version: scl enable devtoolset-8 bash
- TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"


## Experiments
All experiments are conducted on Slurm. Please refer to slurm syntax for header configuration.
### Pre-training
- RAFT
```
sbatch ./scripts/pretrain/raft.sh
```

- Vanilla BERT

```
bash ./scripts/pretrain/vanilla_BERT.sh
```

### Fine-Tuning
- GLUE
    - Low-data
        - RAFT
        ```
        bash ./scripts/finetune/full-finetune/glue-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/full-finetune/glue-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        bash ./scripts/finetune/full-finetune/glue-raft-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/full-finetune/glue-vanilla-array.sh
        ```
        
- SQuAD
    - Low-data
        - RAFT
        ```
        bash ./scripts/finetune/full-finetune/squad-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/full-finetune/squad-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        bash ./scripts/finetune/full-finetune/squad-raft-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/full-finetune/squad-vanilla-array.sh
        ```
- BitFit
    - Low-data
        - RAFT
        ```
        bash ./scripts/finetune/bitfit/glue-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/bitfit/glue-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        bash ./scripts/finetune/bitfit/glue-raft-array.sh
        ```
        - Vanilla BERT
        ```
        bash ./scripts/finetune/bitfit/glue-vanilla-array.sh
        ```
    

### Zero-shot learning
One needs to convert TriviaQA to the SQuAD format using `./triviaqa/convert_to_squad_format.py` first.
- TriviaQA
    - RAFT
        ```
        bash ./scripts/zs/zs-raft-qa.sh
        ```
    - Vanilla BERT
        ```
        bash ./scripts/zs/zs-vanilla-qa.sh
        ```
- SNLI
    - RAFT
        ```
        bash ./scripts/zs/zs-raft-qa.sh
        ```
    - Vanilla BERT
        ```
        bash ./scripts/zs/zs-vanilla-qa.sh
        ```


### Plot 
To plot shapes of rational activation functions in different layers, use `python run_plot.py`.
