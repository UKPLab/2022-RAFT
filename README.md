## Rational BERT

### Structure
```
    ├── args.py
    ├── callbacks.py
    ├── data
    ├── distiller.py              // for KD
    ├── logs                        
    ├── outputs                   //the directory to save model
    ├── pruning_analysis.py       //analyse pruning position
    ├── pruning_utils.py
    ├── README.md
    ├── RecAdam.py                //RecAdam optimizer
    ├── requirements.txt        
    ├── run_distillation.py       //entrypoint for KD
    ├── run_glue.py               //entrypoint for Glue
    ├── run_glue_with_Redadam.py
    ├── run_mlm.py                //entrypoint for pretraining roberta
    ├── scripts                   //slurm scripts
    ├── transformers_modified     //modified transformers library
    
```
This repository includes pretraining, pruning and knowledge ditallation.  

dfsa
### Dependency
- `python
pip install -r requirements.txt
`

- Rational Functions 
Please look into https://github.com/ml-research/rational_activations. If you want to install cuda acceleration version, set `export "FORCE_CUDA"=1`

### Usage
Please check `./scripts/`




