import argparse
from datetime import datetime
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", local_files_only=True)
    return tokenizer, model


def run_model_and_sample(tokenizer, model, prompts, max_new_tokens=96, temperature=1.0, top_k=None, top_p=None, start_record_index=0, timestamp=None):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(model.device)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    generated_tokens = []
    token_list = []
    prob_list = []
    for index in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        
        if top_k is not None or top_p is not None:
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                indices_to_remove = probs < torch.min(top_k_probs, dim=-1, keepdim=True)[0]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        token_text = tokenizer.decode(next_token.item(), skip_special_tokens=False)
        # print("No.", index, "token_text:", token_text)
        if index >= start_record_index:
            token_list.append(token_text)
            prob_list.append(torch.max(probs).item())
        generated_tokens.append(next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        ], dim=-1)
        if tokenizer.eos_token_id is not None:
            if (next_token == tokenizer.eos_token_id).any():
                break

    generated_ids = torch.cat(generated_tokens, dim=-1)
    generated_texts = []
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_texts.append(generated_text)
    data = {
        "tokens": token_list,
        "probs": prob_list,
        "length": len(token_list),
    }
    
    return {
        'generated_ids': generated_ids,
        'generated_texts': generated_texts,
        'data': data
    }


def main():
    parser = argparse.ArgumentParser(description='If no --top_k or --top_p, use greedy search. Single sequence allowed only.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=384)
    parser.add_argument('--prompt_file', type=str, default='prompts/prompt_single.txt',
                        help='File containing the prompt')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-K sampling parameter')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-P (nucleus) sampling parameter')
    parser.add_argument('--dump_json', action='store_true',
                        help='Dump json file')
    parser.add_argument('--start_record_index', type=int, default=0,
                        help='Start recording index')
    parser.add_argument('--save_attn_score_path', type=str, default=None,
                        help='Save attention score tensor')
    
    
    args = parser.parse_args()
    
    with open(args.prompt_file, 'r', encoding='utf-8') as file:
        prompt = file.read()
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tokenizer, model = load_model(args.model_path)
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        # prompt = prompt.replace("2025-08-26", "2025-08-25")
        print("Input:\n", prompt)
        prompts = [prompt]
        result = run_model_and_sample(
            tokenizer, 
            model, 
            prompts, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            start_record_index=args.start_record_index,
            timestamp=timestamp
        )
        print("Generated text:\n", result['generated_texts'][0])
        if args.dump_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = result['data']
            data['output_id'] = timestamp
            data['full_sequence'] = prompt + result['generated_texts'][0]
            if args.top_k is None and args.top_p is None:
                sampling_strategy = "greedy"
            elif args.top_k is not None:
                sampling_strategy = f"topk{args.top_k}"
                if args.top_p is not None:
                    sampling_strategy += f"_topp{args.top_p}"
            else:
                sampling_strategy = f"topp{args.top_p}"
            json_path = os.path.join("jsons", f"{timestamp}_{sampling_strategy}_probs.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        if args.save_attn_score_path:
            all_scores = []
            for i in range(24):
                score = model.model.layers[i].self_attn.sum_score
                all_scores.append(score)
            scores_tensor = torch.stack(all_scores)
            torch.save(scores_tensor, args.save_attn_score_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()