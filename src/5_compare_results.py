import pandas as pd
import numpy as np
import pickle

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

  if ret["causal_problem"] == [0, 1]:
    for key in gold.keys():
      ret[key] = [0, 1]
    return ret

  for key in gold.keys():
    if key == "causal_problem":
      continue
    if key not in ["condition", "response"]:
      if gold[key][0] == predict[key]:
        ret[key] = [1, 1]
      else:
        ret[key] = [0, 1]
    elif key in ["response"]:
      if gold["response"][0] == predict["outcome"]:
        ret[key] = [1, 1]
      else:
        ret[key] = [0, 1]
    else:
      if "condition_variable" in predict.keys():
        if (gold["condition"][0][0] == predict["condition_variable"]) and (gold["condition"][0][1] == predict["condition_value"]):
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
      if gold[key] == predict[key]:
        ret[key] = [1, 1]
      else:
        ret[key] = [0, 1]
    else:
      ret[key] = [0, 1]
  if ret["causal_problem"] == [0, 1]:
    for key in ret.keys():
      ret[key] = [0, 1]
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


if __name__ in "__main__":
  data_gpt_35 = pd.read_csv("../data/run_files/gpt4t_guided_run_p30.csv")
  with open("../data/replicates/evalutate_30p_s1.pkl", "rb") as handle:
    data_s1 = pickle.load(handle)
  print(data_s1)
  data_gpt_35["predict"] = data_s1
  ret = check_data_ours(data_gpt_35)
  for key in ret.keys():
    print(key, ret[key][0] / ret[key][1])
  print(ret)

  