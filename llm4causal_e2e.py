import os
import sys
import argparse

import pandas as pd
import numpy as np
import gc 
import os 
import sys 
import threading 
import numpy as np 
import psutil 
import torch 
import pickle
from accelerate import Accelerator 
from datasets import load_dataset 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel

from src.utilities.e2e_utl import *
from src.utilities.step2_functions import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="path to fine-tuned model",
        default = "model/llama2_v1_mixed_e1_r16"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        help="path to data inputs",
        default = "data_work/"
    )
    parser.add_argument(
        "--temp_path", 
        type=str, 
        help="path to temp csv file",
        default = "data_work/temp.csv"
    )

    args = parser.parse_args()
    return args

def model_prep(args):
    #acc
    accelerator = Accelerator()
    model_name_or_path = "meta-llama/Llama-2-7b-hf" 

    # peft(LoRA)
    peft_model_id = args.model_path #"model/llama2_v1_mixed_e1_r16"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, peft_model_id)

    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

    return model, tokenizer

def process_input(str_in):
    if str_in == "" or str_in == "1":
        # CSL example
        str_in = "Are there any discernible effects on median home price and vacancy rates within the housing_market.csv dataset?"
    elif str_in == "2":
        # ATE example
        str_in = "In the context of the political_engagement.csv, what is the magnitude of influence that party membership (member_party) status exerts on the count of legislation passed (bills_passed)?"
    elif str_in == "3":
        # HTE example
        str_in = "Can we see noticeable shifts in vaccination rates (vaccination_rate) as a result of varying disease incidence, when examining the data for a group with an life expectancy (average_lifespan) of 0.77, according to the public_health.csv?"
    elif str_in == "4":
        # MA example
        str_in = " When analyzing the retail_sales.csv, what role does consumer spending (expenditure) play in mediating how product demand translates into changes in retail revenue?"
    elif str_in == "5":
        # CPL example
        str_in = "With the consumer_electronics.csv data at hand, and acknowledging an innovation rate of 0.69 (innovation_level=0.69), what product releases strategy might be optimal for maximizing sales volume (units_sold)?"
    return str_in

def create_temp_csv(args, input_string, input_type):
    df_dict = {
        "input" : [input_string],
        "output": [""],
        "type": [input_type]
    }
    df = pd.DataFrame(df_dict)
    df.to_csv(args.temp_path)


def preprocess_function1(examples):
    str_use = []
    for x, t in zip(examples["input"], examples["type"]):
        if t == "step1":
            str_c = "Causal Question: " + x + " Structure Response: "
        else:
            str_c = "Function Output: " + x + " interpretation: "
        str_use.append(str_c)
    return tokenizer(
        str_use, 
        padding = "max_length",
        max_length = max_length,
        truncation=True,
        return_token_type_ids = False
    )
    
def run_eval_on_file(args, model):
    # tokenize
    dataset = load_dataset("csv" ,data_files = args.temp_path)
    with accelerator.main_process_first():
        test_dataset = dataset["train"].map(
            preprocess_function1, 
            batched = True,
            num_proc = 1,
            remove_columns = dataset["train"].column_names,
            load_from_cache_file=True,
            desc = "Running Tokenizer on test dataset"
        )
    accelerator.wait_for_everyone()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn = data_collator, batch_size = 1, pin_memory = True
    )
    
    try:
        model, test_dataloader = accelerator.prepare(
            model, test_dataloader
        )
    except:
        test_dataloader = accelerator.prepare(
            test_dataloader
        )
        
    
    model.eval()
    eval_preds = []
    with TorchTracemalloc() as tracemalloc:

        for _, batch in enumerate(tqdm(test_dataloader)):

            batch = {k:v.to("cuda") for k, v in batch.items() if k!= "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus = is_ds_zero_3, max_new_tokens = 200
                )
            outputs = accelerator.pad_across_processes(outputs, dim = 1, pad_index = tokenizer.pad_token_id)
            preds = accelerator.gather_for_metrics(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            preds1 = tokenizer.batch_decode(preds, skip_special_token = True)
            accelerator.print(preds1)
            eval_preds.extend(preds1)
                
    accelerator.wait_for_everyone()
    return eval_preds[0].split("</s>")[0]
        
if __name__ == '__main__':
    # prase arguments
    args = parse_args()
    accelerator = Accelerator()
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3 
    
    # setting constants
    lr = 3e-4 
    seed = 0 
    max_length = 512 #128 
    do_test = False 
    set_seed(seed)
    
    # prepare model & tokenizer
    model, tokenizer = model_prep(args)
    
    # awaiting new inputs
    eval_string = ""
    while(eval_string != "exit"):  
        # input
        eval_string = process_input(input("Please type your question in the command line. Hit enter directly or pass number 1 - 5 for demo questions. Enter 'exit' to quit."))
        if eval_string == "exit":
            break
        
        # step 1
        print("\n\n", "-" * 100)
        print("working on step 1 w/ input:", eval_string)
        create_temp_csv(args, eval_string, "step1")
        step1_output = run_eval_on_file(args, model)
        
        # step 2
        print("\n\n", "-" * 100)
        print("working on step 2 w/ input:", step1_output)
        step2_num_output, tmp_data = route_json_to_function(step1_output, args.data_path)
        temp_output = generate_function_outcome(step1_output, step2_num_output, tmp_data)
        step3_template = generate_interpretation(step1_output, step1_output, temp_output)

        # step 3
        print("\n\n", "-" * 100)
        print("working on step 3 w/ input:", step3_template)
        create_temp_csv(args, step3_template, "step3")
        output = run_eval_on_file(args, model)
        
        # output
        print("-" * 100, "\nSummary:")
        print("input:", eval_string)
        print("step1 result:", step1_output)
        print("step2 result:", step2_num_output)
        print("step3 result:", output)
        print("-" * 100)
