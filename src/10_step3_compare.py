import pandas as pd
import numpy as np 
import matplotlib
import os

def add_tasks(row):
	task = ""
	if row["prompt4interpret"].find('causal structure learning') > -1:
		task = "CSL"
	elif row["prompt4interpret"].find('average treatment effect') > -1:
		task = "ATE"
	elif row["prompt4interpret"].find('heterogeneous treatment effect') > -1:
		task = "HTE"
	elif row["prompt4interpret"].find('mediation analysis') > -1:
		task = "MA"
	elif row["prompt4interpret"].find('causal policy learning') > -1:
		task = "CPL"

	if task == "":
		print("error", row)
		return ""
	else:
		return task
	 
if __name__ in "__main__":
	files = [x for x in os.listdir() 
		if (x.find("check_part1_") > -1) or (x.find("check_part2_") > -1)]
	print(files)

	gold = pd.read_csv("p30_merge_together.csv")
	gold = gold[[gold.columns[x] for x in [4, 3, 2, 5]]]
	print(gold.columns)

	scores = []
	for file in files:
		curr = pd.read_csv(file)
		try:
			curr.columns = ["index", "prompt4interpret", "interpret", "hallu", "incom", "non-flu"]
		except:
			print("cannot infer the file", file)
			print(curr.columns)

		curr_add_model = pd.merge(curr, gold, how = "left", on = ["index", "prompt4interpret", "interpret"])
		curr_add_model["task"] = curr_add_model.apply(add_tasks, axis = 1)
		scores.append(curr_add_model)

	for idx, df_tmp in enumerate(scores):
		score = df_tmp[["model", "hallu", "incom", "non-flu", "task"]]
		#print(files[idx], len(scores[idx]), scores[idx].describe())
		#print(score[["model", "hallu", "incom", "non-flu"]].groupby("model").mean())
		#print("\n", score.groupby(["task", "model"]).mean())

	score = pd.concat(scores)[["model", "hallu", "incom", "non-flu", "task"]]
	print(score[["model", "hallu", "incom", "non-flu"]].groupby("model").mean())
	print(score.groupby(["task", "model"]).mean())
	#print(score.groupby(["model"]).mean())
