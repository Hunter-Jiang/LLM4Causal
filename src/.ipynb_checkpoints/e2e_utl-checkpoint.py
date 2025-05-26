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

def generate_problem(JSON_in):
    if JSON_in.find("</s>") > -1:
        JSON_in = JSON_in.split("</s>")[0]
    JSON_eval = eval(JSON_in)
    if JSON_eval["causal_problem"] == ['CSL', None]:
        problem, method = "causal structure learning", "PC algorithm" 
    elif JSON_eval["causal_problem"] == ['CEL', 'ATE']:
        problem, method = "average treatment effect", "doubly robust estimator"
    elif JSON_eval["causal_problem"] == ['CEL', 'HTE']:
        problem, method = "heterogeneous treatment effect", "S-learner"
    elif JSON_eval["causal_problem"] == ['CEL', 'MA']:
        problem, method = "mediation analysis", "doubly robust estimator"
    elif JSON_eval["causal_problem"] == ['CPL', None]:
        problem, method = "causal policy learning", "Q learning"
    else:
        raise NotImplementedError
    return problem, method

def generate_interpretation(query, JSON_in, function_out):
    problem, method = generate_problem(JSON_in)
    if problem == "mediation analysis":
        n_sentences = 4
    else:
        n_sentences = 3
    prompt =(
      f"(A) is a list of information that includes i) the original causal problem, ii) the class identification of the causal problem, iii) the used method, and iv) the outcomes.\n"
      f"Interpret the results in (A) in response to the original causal problem, using neutral language to paraphrase it more fluently and engagingly.\n"
      f"The output summary is (I).\n"
      f"Guidelines:\n"
      f"1: (I) must concentrate on interpreting the result provided in (A) in response to the problem.\n"
      f"2: (I) must include all the results, methods, and dataset name in (A).\n"
      f"3: (I) may include jargon from (A), but it should not include any other technical terms not mentioned in (A).\n"
      f"4: The problem in (A) is a causal problem, thus (I) should not interpret the results as correlation or association.\n"
      f"5: (I) should use a diversified sentence structure that is also reader-friendly and concise, rather than listing information one by one.\n"
      f"6: Instead of including the problems, (I) should use the original problem to develop a more informative interpretation of the result.\n"
      f"7: (I) has to avoid using strong qualifiers such as 'significant'.\n"
      f"8: (I) has to be {n_sentences} sentences or less long, with no repetition of contents.\n"
      f"9: (I) must not comment on the results.\n"
      f"(A):\n"
      f"i) original causal problem: {query}\n"
      f"ii) class identification of the causal problem: {problem}\n"
      f"iii) used method: {method}\n"
      f"iv) outcomes: {function_out}\n"
      f"(I):"
    )
    return prompt