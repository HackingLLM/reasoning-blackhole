# Repetition pattern researches on GPT-OSS-20B

## Settings
```
NVIDIA GeForce RTX 4090
NVIDIA-SMI 570.86.10
Driver Version: 570.86.10
CUDA Version: 12.8 
Python: 3.10
transformers 4.55.1 (modified)
torch: 2.8.0
kernels: 0.9.0
```

## Preparations
```bash
# inside this repo
git clone https://github.com/huggingface/transformers.git
cp ./record-attn-scores-before-softmax.patch transformers
cd transformers
git am -3 record-attn-scores-before-softmax.patch
pip install -e .
```

## Run
1.
Run 100 attacking prompts and 100 relatively safe prompts using **greedy** search. (with harmony format)

```bash
python infer.py --model_path /path/to/model --max_new_tokens 384 --prompts_file prompts_harmful.txt
```

```bash
python infer.py --model_path /path/to/model --max_new_tokens 384 --prompts_file prompts_safe.txt
```

Our results are saved in `100_results` folder. Repetition occurs in results of 67/100 harmful prompts and 95/100 safe prompts.

2.

Pick any one of the repetition prompts for further research. Put it into `prompts/prompt_single.txt`. 

Remember to modify the paths below.

```bash
# run greedy for token probs and attn scores
python auto_regressive.py --model_path /path/to/model --max_new_tokens 384 --dump_json --save_attn_score_path tensors/attn_scores.pt

# analyse probs
python plot_loop.py --json_path /path/to/json/file

# analyse attention
python attention_loop.py --attn_score_path tensors/attn_scores.pt --json_path /path/to/json/file --model_path /path/to/model
```

## Other interesting findings

We got two log files in directory `./interesting_findings`.

One indicates even one day's difference in chat template leads to an endless loop.

Another indicates that error due to batch inference can leads to an endless loop compared to single prompt inference.