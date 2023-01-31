## Transformers with learnable activation functions
This repository incldues code and model for using rational activation functions, a learnable activation function, in Transformers.

Further details could be found in [Transformers with learnale activation functions](https://arxiv.org/abs/2208.14111)

> Abstract: Activation functions can have a significant impact on reducing the topological complexity of input data and therefore improve the performance of the model. Selecting a suitable activation function is an essential step in neural model design. However, the choice of activation function is seldom discussed or explored in Transformer-based language models. Their activation functions are chosen beforehand and then remain fixed from pre-training to fine-tuning. As a result, the inductive biases they imposed on models cannot be adjusted during this long life cycle. Moreover, subsequently developed models (e.g., RoBERTa, BART, and GPT-3) often follow up prior work (e.g., BERT) to use the same activation function without justification. In this paper, we investigate the effectiveness of using Rational Activation Function (RAF), a learnable activation function, in the Transformer architecture. In contrast to conventional, predefined activation functions, RAFs can adaptively learn optimal activation functions during training according to input data. Our experiments show the RAF-based Transformer (RAFT) achieves a lower validation perplexity than a vanilla BERT with the GELU function. We further evaluate RAFT on downstream tasks in low- and full-data settings. Our results show that RAFT outperforms the counterpart model across the majority of tasks and settings. For instance, RAFT outperforms vanilla BERT on the GLUE benchmark by 5.71 points on average in low-data scenario (where 100 training examples are available) and by 2.05 points on SQuAD in full-data setting. Analysis of the shapes of learned RAFs further unveils that they substantially vary between different layers of the pre-trained model and mostly look very different from conventional activation functions. RAFT opens a new research direction for analyzing and interpreting pre-trained models according to the learned activation functions.

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
- gcc version: scl enable devtoolset-8 sbatch
- TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"


## Experiments
All experiments are conducted on Slurm. Please refer to slurm syntax for header configuration.
### Pre-training
You can download the pertrained model here:

- [Pretrained models](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3719)

Or you can pretrain it by yourself since it only takes less than one day to finish the pretraining stage.

- RAFT
```
sbatch ./scripts/pretrain/raft.sh
```

- Vanilla BERT

```
sbatch ./scripts/pretrain/vanilla_BERT.sh
```

### Fine-Tuning
- GLUE
    - Low-data
        - RAFT
        ```
        sbatch ./scripts/finetune/full-finetune/glue-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/full-finetune/glue-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        sbatch ./scripts/finetune/full-finetune/glue-raft-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/full-finetune/glue-vanilla-array.sh
        ```
        
- SQuAD
    - Low-data
        - RAFT
        ```
        sbatch ./scripts/finetune/full-finetune/squad-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/full-finetune/squad-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        sbatch ./scripts/finetune/full-finetune/squad-raft-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/full-finetune/squad-vanilla-array.sh
        ```
- BitFit
    - Low-data
        - RAFT
        ```
        sbatch ./scripts/finetune/bitfit/glue-raft-lr-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/bitfit/glue-vanilla-lr-array.sh
        ```
    - Full-data
        - RAFT
        ```
        sbatch ./scripts/finetune/bitfit/glue-raft-array.sh
        ```
        - Vanilla BERT
        ```
        sbatch ./scripts/finetune/bitfit/glue-vanilla-array.sh
        ```
    

### Zero-shot learning
One needs to convert TriviaQA to the SQuAD format using `./triviaqa/convert_to_squad_format.py` first.
- TriviaQA
    - RAFT
        ```
        sbatch ./scripts/zs/zs-raft-qa.sh
        ```
    - Vanilla BERT
        ```
        sbatch ./scripts/zs/zs-vanilla-qa.sh
        ```
- SNLI
    - RAFT
        ```
        sbatch ./scripts/zs/zs-raft-qa.sh
        ```
    - Vanilla BERT
        ```
        sbatch ./scripts/zs/zs-vanilla-qa.sh
        ```

### Plot 
To plot shapes of rational activation functions in different layers, use `python run_plot.py`.

### Citation
Please use the following citation:
```
@article{fang2022transformers,
  title={Transformers with Learnable Activation Functions},
  author={Fang, Haishuo and Lee, Ji-Ung and Moosavi, Nafise Sadat and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2208.14111},
  year={2022}
}
```

