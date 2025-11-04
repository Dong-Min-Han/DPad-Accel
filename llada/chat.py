# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import argparse

import generate_baseline
import generate_dropout
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
import time

def chat(args):
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # model = LLaDAModelLM.from_pretrained("/home/dh783/AQLM/converted-llada-8b-instruct-2bit", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval() 
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = 512
    steps = 512
    block_length = 32
    window = 128
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        torch.manual_seed(42) # fix the random seed

        # user_input = input("Enter your question: ") # uncomment here if you actually want to chat

        # user_input = "Translate the following English sentence to French: 'The quick brown fox jumps over the lazy dog.'"
        # user_input = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        # user_input = "Explain the theory of relativity in simple terms."
        # user_input = "Explain Quantum Computing in simple terms."
        user_input = "Alice is taller than Bob, and Bob is taller than Carol. Who is the tallest?"
        user_input = "Write a Python function that removes duplicates from a list while preserving order."

        if user_input == "exit": break

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        
        dropout = 'gaussian' # use dpad or not
        # dropout = None

        threshold = 0.9 # use fast-dllm or not
        # threshold = None

        use_cache = False # turn on prefix cache
        if_cache_position = False # decide dual cache or prefix cache
        
        start_time = time.time()
        if dropout is not None:
            if use_cache:
                if if_cache_position:
                    out, nfe = generate_dropout.generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold, dropout='gaussian', sigma=4, scale=2, window=window)
                else:
                    out, nfe = generate_dropout.generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold, dropout='gaussian', sigma=4, scale=2, window=window)
            else:
                out, nfe = generate_dropout.generate(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold, dropout='gaussian', sigma=4, scale=2, window=window)        
        else:
            if use_cache:
                if if_cache_position:
                    out, nfe = generate_baseline.generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)
                else:
                    out, nfe = generate_baseline.generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)
            else:
                out, nfe = generate_baseline.generate(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)        
        end_time = time.time()
        execution_time = end_time - start_time
        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}\n")
        print(f'dropout: {dropout}, threshold: {threshold}, use_cache: {use_cache}, use_dual_cache: {if_cache_position}')
        print(f"Number of actual forward passes: {nfe}")
        print(f"Elapsed: {execution_time:.2f} s")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')
        break # commment here out if you actually want to chat.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    chat(args)