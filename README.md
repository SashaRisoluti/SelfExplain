# SelfExplain Framework

The code for the SelfExplain framework (https://arxiv.org/pdf/2103.12279.pdf) 

## Preprocessing

Data for preprocessing available in `data/` folder

On a python shell, do the following for installing the parser

```python
>>> import benepar
>>> benepar.download('benepar_en3')
```

```shell
sh scripts/run_preprocessing.sh
```


## Training

```shell
sh scripts/run_self_explain.sh
```
## Generation (Inference)

(In Progress)

```sh
 python model/infer_model.py
        --ckpt $PATH_TO_BEST_DEV_CHECKPOINT \
        --concept_map $DATA_FOLDER/concept_idx.json \ 
        --batch_size $BS \
        --paths_output_loc $PATH_TO_OUTPUT_PREDS
 ```

## Demo 

Coming Soon ... 

## Citation 

```
@misc{rajagopal2021selfexplain,
      title={SelfExplain: A Self-Explaining Architecture for Neural Text Classifiers}, 
      author={Dheeraj Rajagopal and Vidhisha Balachandran and Eduard Hovy and Yulia Tsvetkov},
      year={2021},
      eprint={2103.12279},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```