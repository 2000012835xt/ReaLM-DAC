
import torch
import os, gc
import numpy as np
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
# from err_inject_msd import W8A8Linear, NoisyW8A8Linear,W8A8MatMul,NoisyW8A8MatMul
from smoothquant.error_inject import W8A8Linear, NoisyW8A8Linear, W8A8MatMul, NoisyW8A8MatMul
from sampling import autoregressive_sampling
from datasets import load_dataset


from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
import pdb
from tqdm import tqdm
import time

import re
import argparse
import jsonlines
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import LlamaTokenizer

def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
            m.matmul2 = W8A8MatMul(act_quant=act_quant,quantize_output=True)
    return model

def quantize_error_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True,err_prob=0
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    i = 0
    for name, m in model.model.named_modules():
        #print(name)
        if isinstance(m,(LlamaMLP,MistralMLP)):
            if i==0:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                # m.gate_proj = NoisyW8A8Linear.from_float(
                #     m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                # )
                # m.up_proj = NoisyW8A8Linear.from_float(
                #     m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                # )
                # m.down_proj = NoisyW8A8Linear.from_float(
                #     m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                # )
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                #i += 1               
        elif isinstance(m, (LlamaAttention,MistralAttention)):
            if i==0:
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                # m.q_proj = NoisyW8A8Linear.from_float(
                #     m.q_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                #     err_prob=err_prob
                # )
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                # m.k_proj = NoisyW8A8Linear.from_float(
                #     m.k_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                #     err_prob=err_prob
                # )
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                # m.v_proj = NoisyW8A8Linear.from_float(
                #     m.v_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                #     err_prob=err_prob
                # )
                m.o_proj = W8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                )
                # m.o_proj = NoisyW8A8Linear.from_float(
                #     m.o_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob
                # )
                m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
                # m.matmul1 = NoisyW8A8MatMul(act_quant=act_quant, quantize_output=False,err_prob=err_prob)
                
                # m.matmul2 = W8A8MatMul(act_quant=act_quant,quantize_output=True)
                m.matmul2 = NoisyW8A8MatMul(act_quant=act_quant, quantize_output=True,err_prob=err_prob)
            else:
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )  
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                ) 
                m.o_proj = W8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                )
                m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
                m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)                                 
    return model


# origin code from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

ans_re1 = re.compile(r"(\-?[0-9][0-9\.\,]*)")

ans_re2 = re.compile(r'=\s*(\$?-?[0-9][0-9\.\,]*)')

prefix_sky1 = 'answer is'
prefix_sky2 = '答案是'

INVALID_ANS = "[invalid]"

def get_match_str(match, idx):
    match_str = match[idx]
    match_str = match_str.replace(",", "")
    if match_str.endswith('.'):
        match_str = match_str[:-1]
    if match_str.endswith('.00'):
        match_str = match_str[:-3]
    if match_str.endswith('.0'):
        match_str = match_str[:-2]
    return match_str

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sents.append(sent)
    return sents

def generate_sample(model, model_decode, tokenizer, input_txt):
    input_ids = tokenizer([input_txt], padding=False)["input_ids"]
    context_enc = torch.tensor(input_ids, device=model.device)
    input_token_len = len(input_ids[0])
    # pdb.set_trace()
    num_tokens=150
    top_k = 1
    top_p = 0.
    output_ids=autoregressive_sampling(x=context_enc,model=model,model_decode=model_decode, N=num_tokens,temperature=1, top_k=top_k, top_p=top_p)
    output_text=tokenizer.decode(output_ids[0,input_token_len:],skip_special_tokens=True)
    return output_text

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def extract_answer(text):
    if prefix_sky1 in text:
        text = text.split(prefix_sky1)[-1]
    if prefix_sky2 in text:
        text = text.split(prefix_sky2)[-1]
    match1 = re.findall(ans_re1, text)
    match2 = re.findall(ans_re2, text)
    ans = []
    if match1:
        match_str1 = get_match_str(match1, -1)
        ans.append(match_str1)
    if match2:
        match_str2 = get_match_str(match2, -1).replace('$','')
        ans.append(match_str2)
    if len(ans) > 0:
        return eval(ans[-1])
    else:
        return INVALID_ANS

def is_correct(completion, answer):
    completion = completion.split('</s>')[0]
    completion = completion.split('\n\n\n')[0]
    completion = completion.split("\n\n")[0]
    completion = completion.split("Question:")[0]

    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    try:
        clear_answer = extract_answer(completion)
    except Exception as error:
        print(f"Can't extracr answer correctly:{error}")
        clear_answer = None
    return clear_answer == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="Llama_gsm8k_res.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("./gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"].select(range(300))
    # err_prob_list=[0.0, 1e-8,1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    err_prob_list=[0.0, 1e-8,1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # err_prob_list = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    gsm8k_acc_list = []
    start_time_all= time.time()
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", trust_remote_code=True, padding_side='left'
    )
    if "qwen" in "meta-llama/Llama-2-7b-hf".lower():
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.eos_token = '<|endoftext|>'
    else:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    for i in range(len(err_prob_list)):
        err_prob=err_prob_list[i]
        print('matmul2_two')
        print(err_prob)
        time_start=time.time()
        print("Loading model ...")
        # Llama2_model_1 = AutoModelForCausalLM.from_pretrained(
        #     "meta-llama/Llama-2-7b-hf", device_map="auto", trust_remote_code=True
        # ).eval()
        Llama2_model_2 = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", trust_remote_code=True
        ).eval()
        act_scales = torch.load('act_scales/llama-2-7b.pt')
        # smooth_lm(Llama2_model_1, act_scales, 0.85)
        smooth_lm(Llama2_model_2, act_scales, 0.85)

        # normal_Llama2 = quantize_model(Llama2_model)
        # normal_Llama2.generation_config = GenerationConfig.from_pretrained(
        #     "meta-llama/Llama-2-7b-hf", trust_remote_code=True
        # )
        # normal_Llama2.generation_config.do_sample = False
        # normal_Llama2 = quantize_model(Llama2_model_1)
        error_Llama2 = quantize_error_model(Llama2_model_2, err_prob=err_prob)

        del Llama2_model_2#, Llama2_model_1, 
        gc.collect()
        # error_Llama2.generation_config = GenerationConfig.from_pretrained(
        #     "meta-llama/Llama-2-7b-hf", trust_remote_code=True
        # )
        # error_Llama2.generation_config.do_sample = False

        f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
        tot_length = test.num_rows
        acc_res = []
        for doc in tqdm(test,desc='evaluating'):
            #pdb.set_trace()
            context = doc_to_text(doc)
            completion = generate_sample(error_Llama2, error_Llama2, tokenizer, context)
            #completion = generate_sample(model,tokenizer,context)
            answer = doc["answer"]
            acc = is_correct(completion, answer)
            doc["completion"] = completion
            doc["acc"] = acc
            f_output.write(doc)
            acc_res.append(acc)

        f_output.close()
        print(acc_res)
        gsm8k_acc = np.mean(acc_res)
        print("Acc: ", gsm8k_acc)
        gsm8k_acc_list.append(gsm8k_acc)

        #删除当前循环的模型和配置文件
        # normal_Llama2.cpu()
        error_Llama2.cpu()
        del error_Llama2#, normal_Llama2
        torch.cuda.empty_cache()

        time_end = time.time()
        time_i = time_end-time_start
        print('time_i',time_i/60)
    for item in gsm8k_acc_list:
        print(item)
    end_time_all=time.time()
    time_all=end_time_all - start_time_all
    print('time_all', time_all/60)
    

