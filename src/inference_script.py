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


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

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
    model_name_or_path = "meta-llama/Llama-2-7b-hf" 
    #dataset_name = "eli5"
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3 
    peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode = False, r = 2, 
                            lora_alpha = 32, lora_dropout = 0.1,
                            target_modules = ["q_proj", "v_proj"]) #["query_key_value"]) #
    lr = 3e-4 
    num_epochs = 1
    batch_size = 1
    seed = 0 
    max_length = 512 #128 
    do_test = False 
    set_seed(seed)
    # no need to specifiy read on the miain proces
    dataset = load_dataset("csv" ,data_files = "data/v2_30p_mixed_data/interpret_ready_go_v2.csv") #"v1_300p_data/test_p300.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  
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
        test_dataset, collate_fn = data_collator, batch_size = batch_size, pin_memory = True
    )
    
    print(next(iter(test_dataloader)))
    
    peft_model_id = "model/llama2_v1_mixed_e1_r16"

    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
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
    with open("data/v2_30p_mixed_data/output/inter_e2e_ours_s3_inter.pkl", "wb") as handle: #p300_inference_v1.pkl
        pickle.dump(eval_preds, handle)
        
    #model.save_pretrained("llama2_cd_v1_10e/")
                
if __name__ == "__main__":
    
    main()
