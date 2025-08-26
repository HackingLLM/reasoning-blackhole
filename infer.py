import argparse
import json
import sys
from transformers import AutoTokenizer, GptOssForCausalLM, AutoModelForCausalLM
import torch

def load_model(model_path):
    # use left padding here
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", local_files_only=True)
    # model = GptOssForCausalLM.from_pretrained(model_path, device_map="cuda", local_files_only=True)
    return tokenizer, model


def generate_text(tokenizer, model, prompts, max_new_tokens=96):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    
    results = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(
            output[inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=False
        )
        results.append({
            'prompt': prompts[i],
            'generated': generated_text
        })
    
    return results

def save_expert_activation(model, file_name):
    final_tensor = torch.stack([layer.mlp.counts for layer in model.model.layers])
    # for i in range(len(model.model.layers)):
    #     layer = model.model.layers[i]
    #     print(f"layer {i}:", layer.mlp.counts)
    print(final_tensor)
    torch.save(final_tensor, file_name)

def main():
    parser = argparse.ArgumentParser(description='batched inference')
    # parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--model_path', type=str, 
                       default="/home/gpt-oss/models/gpt-oss-20b")
    parser.add_argument('--max_new_tokens', type=int, default=96)
    parser.add_argument('--prompts_file', type=str, default='prompts.txt',
                        help='File containing prompts, one per line')
    # parser.add_argument('--file_name', type=str, default='prompts.pt')
    
    args = parser.parse_args()
    # with open("configs.json", 'r', encoding='utf-8') as file:
    #     config = json.load(file)
    # base_prompt = config.get("prompt", "Once upon a time, ")
    # prompts = [base_prompt] * args.batch_size

    with open(args.prompts_file, 'r', encoding='utf-8') as file:
        messages = file.readlines()

    try:
        tokenizer, model = load_model(args.model_path)
        prompts = []
        for message in messages:
            message = [{"role": "user", "content": message}]
            input = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(input)

        results = generate_text(tokenizer, model, prompts, args.max_new_tokens)
        for i, result in enumerate(results):
            print(f"Input {i+1}: {result['prompt']}")
            print(f"Ouput {i+1}: {result['generated']}")
            print("-" * 50)
        # print_expert_activation(model, args.file_name)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()