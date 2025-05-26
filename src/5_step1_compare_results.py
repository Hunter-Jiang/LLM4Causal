import pandas as pd
import numpy as np
import pickle
import os

def check_row(row):
  gold = eval(row["output"])
  fun = row["name"]
  predict = eval(row["arguments"])

  ret = {}
  match fun:
    case "causal_graph_learning":
      ret["causal_problem"] = [1, 1] if gold["causal_problem"] == ["CSL", None] else [0, 1]
    case "average_treatment_effect":
      ret["causal_problem"] = [1, 1] if gold["causal_problem"] == ["CEL", "ATE"] else [0, 1]
    case "heterogeneous_treatment_effect":
      ret["causal_problem"] = [1, 1] if gold["causal_problem"] == ["CEL", "HTE"]  else [0, 1]
    case "mediation_analysis":
      ret["causal_problem"] = [1, 1] if gold["causal_problem"] == ["CEL", "MA"]  else [0, 1]
    case "causal_policy_learning":
      ret["causal_problem"] = [1, 1] if gold["causal_problem"] == ["CPL", None] else [0, 1]
    case _:
      ret["causal_problem"] = [0, 1]

  #if ret["causal_problem"] == [0, 1]:
  #  for key in gold.keys():
  #    ret[key] = [0, 1]
  #  return ret

  for key in gold.keys():
    if key == "causal_problem":
      continue
    if key not in ["condition", "response"]:
      if key not in predict:
        ret[key] = [0, 1]
      else:
        if soft_equal(gold[key][0],predict[key]):# gold[key][0] == predict[key]:
          ret[key] = [1, 1]
        else:
          ret[key] = [0, 1]
    elif key in ["response"]:
      if "outcome" in predict:
        if soft_equal(gold["response"][0],predict["outcome"]): #gold["response"][0] == predict["outcome"]:
          ret[key] = [1, 1]
        else:
          ret[key] = [0, 1]
      else:
        ret[key] = [0, 1]
    else:
      if "condition_variable" in predict.keys():
        if soft_equal(
            gold["condition"][0][0],
            predict["condition_variable"]) and soft_equal(
            gold["condition"][0][1],
            predict["condition_value"]):
            #:#(gold["condition"][0][0] == predict["condition_variable"]) and (gold["condition"][0][1] == predict["condition_value"]):
          ret[key] = [1, 1]
        else:
          ret[key] = [0, 1]

  return ret #gold, fun, predict

def check_row_ours(row):
  gold = eval(row["output"])
  try:
    predict = eval(row["predict"].split("</s>")[0])
  except:
    predict = {}
    print(row["predict"])

  ret = {}
  for key in gold.keys():
    if key in predict.keys():
      if soft_equal(gold[key],predict[key]):
        ret[key] = [1, 1]
      else:
        ret[key] = [0, 1]
    else:
      ret[key] = [0, 1]
  #if ret["causal_problem"] == [0, 1]:
  #  for key in ret.keys():
  #    ret[key] = [0, 1]
  return ret

def check_data_ours(data):
  tot_res = {}
  for idx, row in data.iterrows():
    #print(idx)
    res = check_row_ours(row)
    for key in res.keys():
      if key not in tot_res:
        tot_res[key] = res[key]
      else:
        tot_res[key] = [x + y for x, y in zip(tot_res[key], res[key])]

    error_count = 0
    for key in res.keys():
      if res[key] == [0, 1]:
        error_count += 1
    if error_count > 0:
      print(row["input"])
      print(error_count, "error\n gold  :", row["output"], "\npredict:", row["predict"])
      print("-"*50)
  return tot_res

def check_data(data):
  tot_res = {}
  for idx, row in data.iterrows():
    res = check_row(row)
    for key in res.keys():
      if key not in tot_res:
        tot_res[key] = res[key]
      else:
        tot_res[key] = [x + y for x, y in zip(tot_res[key], res[key])]

    error_count = 0
    for key in res.keys():
      if res[key] == [0, 1]:
        error_count += 1
    if error_count > 0:
      print(row["input"])
      print(error_count, "error\n gold  :", row["output"], "\npredict:", str(eval(row["arguments"])))
      print("-"*50)
  return tot_res

def soft_equal(input_a, input_b):
  #print(stra, strb, type(stra), type(strb), "\n\n")
  #print(type(stra[0]))

  # special trt to condition (as it is tuple)
  if type(input_a) == list:
    if type(input_a[0]) == tuple:
      input_a = [input_a[0][0], input_a[0][1]]
      input_b = [input_b[0][0], input_b[0][1]]
    input_a  = [str(x) for x in input_a]
    input_b  = [str(x) for x in input_b]
  else:
    input_a = [str(input_a)]
    input_b = [str(input_b)]

  # soft match
  flag = True
  for item in input_a:
    match_flag = False
    item_soft = item
    # delete "s" and "es"
    if item_soft[-2:] == "es":
      item_soft = item_soft[:-2]
    elif item_soft[-1] == "s":
      item_soft = item_soft[:-1]
    # find inclusion
    for item1 in input_b:
      if item1.find(item_soft) > -1:
        match_flag = True
        break
    if not match_flag:
      flag = False
  #flag = input_a == input_b
  return flag

if __name__ in "__main__":
  # ours == LLM4Causal, others == gpt
  work_type = "ours"
  folder = "../data/run_files/"
  folder_ours = "../data/replicates/"
  file = [x for x in os.listdir(folder) if (x[:6] == "gpt-3_") or (x[:6] == "gpt-4_")]
  file_ours = [x for x in os.listdir(folder_ours) if (x[:5] == "step1") or (x[:5] == "guide")]

  if work_type == "ours":
    output_df = []
    for work_file in file_ours:
      data_gpt_35 = pd.read_csv(folder + file[0]) #_guided
      with open(folder_ours + work_file, "rb") as handle:
        data_s1 = pickle.load(handle)
      if work_file.find("guided") > -1:
        data_s1 = ["{" + x for x in data_s1]
        #print(data_s1[0])
        #break
      data_gpt_35["predict"] = data_s1
      ret = check_data_ours(data_gpt_35)
      ret1 = {}
      for key in ret.keys():
        ret1[key] = ret[key][0] / ret[key][1]
      spl = work_file.split("_")
      ret1["model"] = spl[1]
      ret1["temperature"] = spl[2]
      if work_file.find("guided") > -1:
        ret1["guided"] = "yes"
        ret1["temperature"] = "0"
        ret1["model"] = spl[2]
      else:
        ret1["guided"] = "no"
      output_df.append(ret1)
    all_df = pd.DataFrame(output_df)
    all_df.to_csv(folder + "all_ours_performance_check.csv")
  else:
    output_df = []
    for work_file in file:
      data_gpt_35 = pd.read_csv(folder + work_file) #_guided
      ret = check_data(data_gpt_35)
      ret1 = {}
      for key in ret.keys():
        ret1[key] = ret[key][0] / ret[key][1]
      if work_file[:6] == "gpt-3_":
        ret1["model"] = "gpt-3.5-turbo"
      else:
        ret1["model"] = "gpt-4-turbo"
      if work_file.find("guided") > -1:
        ret1["guided"] = "yes"
      else:
        ret1["guided"] = "no"
      spl = work_file.split("_")
      ret1["run"] = spl[2]
      ret1["temperature"] = spl[3]
      output_df.append(ret1)
    all_df = pd.DataFrame(output_df)
    all_df.to_csv(folder + "all_gpt_performance_check.csv")
    print(all_df.head())



  