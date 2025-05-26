import gc 
import os 
import sys 
import threading 
import numpy as np 
import psutil 
import torch 
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
from peft import LoraConfig, TaskType, get_peft_model

def b2mb(x):
    """
    Bytes to MegaBytes
    """
    return int(x / 2**20)
class TorchTracemalloc:
    def __enter__(self):
        gc.collect() # Clean unreferenced objects
        torch.cuda.empty_cache() # Avoid OOM
        torch.cuda.reset_max_memory_allocated() # reset peak gauge to zero 
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process(os.getpid())
        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True 
        peak_monitor_thread = threading.Thread(target = self.peak_monitor_func)
        peak_monitor_thread.daemon = True 
        peak_monitor_thread.start()
        return self
    def cpu_mem_used(self):
        return self.process.memory_info().rss 
    def peak_monitor_func(self):
        self.cpu_peak = -1 
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break
    def __exit__(self, *exc):
        self.peak_monitoring = False 
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
def main():
    accelerator = Accelerator()
    model_name_or_path = "meta-llama/Llama-2-7b-hf" #"tiiuae/falcon-7b" #
    #dataset_name = "eli5"
    peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode = False, r = 16, 
                            lora_alpha = 32, lora_dropout = 0.1,
                            target_modules = ["q_proj", "v_proj"]) #["query_key_value"]) #
    lr = 3e-4 
    num_epochs = 2
    batch_size = 1
    seed = 0 
    max_length = 128 #128 
    do_test = False 
    set_seed(seed)
    # no need to specifiy read on the miain proces
    dataset = load_dataset("parquet" ,data_files = "train_ver1.parquet")
    dataset = dataset["train"].train_test_split(test_size = 0.01)
    dataset = dataset.flatten()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    def preprocess_function(examples):
        str_use = ["Causal Question: " + x + " Structure Response: " + y for x, y in zip(examples["Input"], examples["Output_parsed"])]
        return tokenizer(
            str_use, 
            padding = "max_length",
            max_length = max_length,
            truncation=True,
            return_token_type_ids = False
        )
    def preprocess_function1(examples):
        str_use = ["Causal Question: " + x + " Structure Response: " for x, y in zip(examples["Input"], examples["Output_parsed"])]
        return tokenizer(
            str_use, 
            padding = "max_length",
            max_length = max_length,
            truncation=True,
            return_token_type_ids = False
        )
    with accelerator.main_process_first():
        train_dataset = dataset["train"].map(
            preprocess_function, 
            batched = True,
            num_proc = 1,
            remove_columns = dataset["train"].column_names,
            load_from_cache_file=True,
            desc = "Running Tokenizer on train dataset"
        )
    with accelerator.main_process_first():
        test_dataset = dataset["test"].map(
            preprocess_function1, 
            batched = True,
            num_proc = 1,
            remove_columns = dataset["test"].column_names,
            load_from_cache_file=True,
            desc = "Running Tokenizer on test dataset"
        )
    accelerator.wait_for_everyone()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle = True, collate_fn = data_collator, batch_size = batch_size, pin_memory = True
    )
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn = data_collator, batch_size = batch_size, pin_memory = True
    )
    
    print(next(iter(train_dataloader)))
    
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = (len(train_dataloader) * num_epochs)
    )
    
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    
    accelerator.print(model)
    
    is_ds_zero_3 = False 
    
    if getattr(accelerator.state, "deepspeed_plugin", None):
        
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3 
        
    for epoch in range(num_epochs):
        
        with TorchTracemalloc() as tracemalloc:
            
            model.train()
            total_loss = 0.0 
            for step, batch in enumerate(tqdm(train_dataloader)):
                
                outputs = model(**batch)
                loss = outputs.loss 
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        
        # pring GPU status 
        accelerator.print("GPU Memory before entering the train: {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end - begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max - begin): {}".format(tracemalloc.peaked))
        accelerator.print("GPU Total Peak Memory consumed during the train (max): {}".format(tracemalloc.peaked + b2mb(tracemalloc.begin)))
        
        # print cpu status 
        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print("CPU Total Peak Memory consumed during the train (max): {}".format(tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)))
        
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        
        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            
            for _, batch in enumerate(tqdm(test_dataloader)):
                
                batch = {k:v for k, v in batch.items() if k!= "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus = is_ds_zero_3, max_new_tokens = 100
                    )
                outputs = accelerator.pad_across_processes(outputs, dim = 1, pad_index = tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                preds1 = tokenizer.batch_decode(preds, skip_special_token = True)
                accelerator.print(preds1)
                eval_preds.extend(preds1)
        
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("llama2_r16_e2_128/", save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
                
if __name__ == "__main__":
    
    main()