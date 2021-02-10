# MAB for LM

## Data

#### Getting the data
Download the WikiText-103 dataset from [here](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

#### Processing the data
* XXXXXX

## Training
To run on school's computers run:

```
python run_clm.py --model_name_or_path gpt2 --train_file="data/raw_data/wikitext-103/wiki-*.valid.tokens.txt" --validation_file="data/raw_data/wikitext-103/wiki-conservatory.valid.tokens.txt" --do_train --do_eval --output_dir /tmp/test-clm --overwrite_cache --num_groups=6 --per_device_train_batch_size=2 --eval_steps=2 --evaluation_strategy=steps --scoring_function=exp3 --fp16 --logging_steps=2 --overwrite_output_dir
```
from `/specific/netapp5_3/rent_public/olab-01-08-2021/kirstain/mab_lm`

### Description
There are k groups (actions) to sample data from and a scoring function, f, that keeps a score for each group.

2) At time t, we sample a group from the induced distribution
3) we do s training steps (we currently support only s=1)
4) we sample b batches from the dev set (we currently support only the entire dev se) and get the cost, c_t, which is the current perplexity minus the last one.
5) we update the scoring function according the cost and return to 1.

### How to run

### TODO
* Support b batches in 4 instead of the entire dev set.
