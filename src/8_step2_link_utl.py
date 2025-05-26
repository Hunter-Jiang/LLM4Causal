import pandas as pd 
import numpy as np
import pickle
import os
from step2_data_gen import *
from step2_functions import *

N = 5000
## function for generating datasets
def create_dataset_for_question(json_in, output_dir, all_vars):
	json_eval = eval(json_in)
	os.makedirs(output_dir, exist_ok = True)
	if json_eval["causal_problem"] == ['CSL', None]:
		if json_eval["nodes"][0] == "all_variables":
			num_var = np.random.randint(low = 2, high = len(all_vars[json_eval["dataset"][0]]))
			nodes = list(np.random.choice(all_vars[json_eval["dataset"][0]], num_var, replace = False))
			#print(nodes)
		else:
			nodes = json_eval["nodes"]
		data_ret = generate_CGL_data(n_data = N, n_var = len(nodes))
		# save data
		df_save = pd.DataFrame(data_ret[1])
		df_save.columns = nodes
		df_save.to_csv(output_dir + json_eval["dataset"][0])
		# save truth
		with open(output_dir + "truth.pkl", "wb") as handle:
			pickle.dump(data_ret[0], handle)

	elif json_eval["causal_problem"] == ['CEL', "ATE"]:
		data_ret = generate_ATE_data(n_data = N, n_x_var = 1)
		# save data
		df_save = pd.DataFrame(data_ret[1])
		columns = ["var_" + str(x) for x in range(len(df_save.columns))]
		columns[0] = json_eval['treatment'][0]
		columns[-1] = json_eval['response'][0]
		df_save.columns = columns
		df_save.to_csv(output_dir + json_eval["dataset"][0])
		# save truth
		get_solution_for_data("ATE", json_eval, data_ret, output_dir + "truth.pkl")
		#with open(output_dir + "truth.pkl", "wb") as handle:
		#	pickle.dump(data_ret[0], handle)
	elif json_eval["causal_problem"] == ['CEL', "HTE"]:
		data_ret = generate_HTE_data(n_data = N, n_x_var = 2)
		# save data
		for idd, dff in enumerate(data_ret[1:]):
			df_save = pd.DataFrame(dff)
			columns = ["var_" + str(x) for x in range(len(df_save.columns))]
			columns[0] = json_eval['treatment'][0]
			columns[-2] = json_eval['condition'][0][0]
			columns[-1] = json_eval['response'][0]
			df_save.columns = columns
			if idd == 0:
				df_save.to_csv(output_dir + json_eval["dataset"][0])
		# save truth
		get_solution_for_data("HTE", json_eval, df_save, output_dir + "truth.pkl")
	elif json_eval["causal_problem"] == ['CEL', "MA"]:
		data_ret = generate_MA_data(n_data = N)
		# save data
		for idd, dff in enumerate(data_ret[1:]):
			df_save = pd.DataFrame(dff)
			columns = ["var_" + str(x) for x in range(len(df_save.columns))]
			columns[0] = json_eval['treatment'][0]
			columns[1] = json_eval['mediator'][0]
			columns[2] = json_eval['response'][0]
			df_save.columns = columns
			if idd == 0:
				df_save.to_csv(output_dir + json_eval["dataset"][0])
		# save truth
		get_solution_for_data("MA", json_eval, df_save, output_dir + "truth.pkl")
	elif json_eval["causal_problem"] == ['CPL', None]:
		# generate data
		data_ret = generate_CPL_data(n_trajectory = int(N / 10), n_data = 2, n_x_var = 1)
		# save data
		for idd, dff in enumerate(data_ret):
			df_save = pd.DataFrame(dff)
			columns = ["var_" + str(x) for x in range(len(df_save.columns))]
			columns[0] = json_eval['treatment'][0] + "-0"
			columns[1] = json_eval['treatment'][0] + "-1"
			columns[2] = json_eval['condition'][0][0] + "-0"
			columns[3] = json_eval['condition'][0][0] + "-1"
			columns[-1] = json_eval['response'][0]
			df_save.columns = columns
			if idd == 0:
				df_save.to_csv(output_dir + json_eval["dataset"][0])
		# save truth
		get_solution_for_data("CPL", json_eval, df_save, output_dir + "truth.pkl")
	return json_eval

## function for saving necessary outputs
def get_solution_for_data(task, json_eval, data_in, save_dir):
	if task == "CSL":
		ret = data_in[0]
	elif task == "ATE":
		ret = data_in[0]
	elif task == "HTE":
		ret = solve_HTE_function(
			data_in, 
			json_eval["treatment"][0], 
			json_eval["response"][0], 
			json_eval["condition"])
	elif task == "MA":
		ret = solve_MA_function(
			data_in, 
			json_eval["treatment"][0], 
			json_eval["response"][0], 
			json_eval["mediator"][0])
	elif task == "CPL":
		ret = solve_CPL_function(
			data_in, 
			json_eval["treatment"][0], 
			json_eval["response"][0], 
			json_eval["condition"][0]
			)
	with open(save_dir, "wb") as handle:
		pickle.dump(ret, handle)


## function for actual solving problems
def route_json_to_function(json_in, saved_dir):
	json_eval = eval(json_in)
	data = pd.read_csv(saved_dir + json_eval["dataset"][0])
	data = data[[x for x in data.columns if x.find("Unnamed") == -1]]
	if json_eval["causal_problem"] == ['CSL', None]:
		ret = solve_CSL_function(data)
		#print("-"* 100, ret)
		with open(saved_dir + "truth.pkl", "rb") as handle:
			true = pickle.load(handle)
		#print("-"* 100, true)
		x = []; [x.extend(y) for y in ret]
		x1 = []; [x1.extend(y) for y in true]
		diff = [y - y1 for y, y1 in zip(x, x1)]
		#print(diff)
		if max([abs(x) for x in diff]) < 0.5:
			print("ok")
		else:
			print(true, "\n", ret)
		with open(saved_dir + "output.pkl", "wb") as handle:
			pickle.dump(ret, handle)

	elif json_eval["causal_problem"] == ['CEL', "ATE"]:
		ret = solve_ATE_function(data, json_eval["treatment"][0], json_eval["response"][0])
		#print(ret)
		with open(saved_dir + "truth.pkl", "rb") as handle:
			true = pickle.load(handle)
		print("estiamte coef:", true)
		if abs(true[0] - ret) < 0.5:
			print("ok")
		else:
			print(true, ret)
		with open(saved_dir + "output.pkl", "wb") as handle:
			pickle.dump(ret, handle)

	elif json_eval["causal_problem"] == ['CEL', "HTE"]:
		ret = solve_HTE_function(data, json_eval["treatment"][0], json_eval["response"][0], json_eval["condition"])
		with open(saved_dir + "truth.pkl", "rb") as handle:
			true = pickle.load(handle)
		if abs(true - ret) < 1:
			print("ok")
		else:
			print(true, ret)
		with open(saved_dir + "output.pkl", "wb") as handle:
			pickle.dump(ret, handle)

	elif json_eval["causal_problem"] == ['CEL', "MA"]:
		ret = solve_MA_function(data, json_eval["treatment"][0], json_eval["response"][0], json_eval["mediator"][0])
		with open(saved_dir + "truth.pkl", "rb") as handle:
			true = pickle.load(handle)
		if abs(true[0] - ret[0]) < 0.5:
			print("ok")
		else:
			print(true, ret)
		with open(saved_dir + "output.pkl", "wb") as handle:
			pickle.dump(ret, handle)

	elif json_eval["causal_problem"] == ['CPL', None]:
		ret = solve_CPL_function(
			data, 
			json_eval["treatment"][0], 
			json_eval["response"][0], 
			json_eval["condition"][0]
			)
		with open(saved_dir + "truth.pkl", "rb") as handle:
			true = pickle.load(handle)
		if abs(true[0] - ret[0]) < 0.5:
			print("ok")
		else:
			print(true, ret)

		with open(saved_dir + "output.pkl", "wb") as handle:
			pickle.dump(ret, handle)
	return json_eval


if __name__ in "__main__":
	## get all samples
	df = pd.read_csv("evaluate_p30.csv")[["input", "output"]]
	#print(df.head())

	## get extra entities
	vars_backup = {}
	entities = pd.read_csv("entity_w_names.csv")
	for idx, row in entities.iterrows():
		data_name = "_".join(row["field"].strip().split(" ")) + ".csv"
		if data_name in vars_backup:
			vars_backup[data_name].append("_".join(row["variable"].split(" ")))
		else:
			vars_backup[data_name] = ["_".join(row["variable"].split(" "))]
	#print(vars_backup)

	for idx, row in df.iloc[90:].iterrows():
		qs = create_dataset_for_question(row["output"], "datafiles/" + str(idx) + "/", vars_backup)
		#print(qs)
		qs1 = route_json_to_function(row["output"], "datafiles/" + str(idx) + "/")
		#print(qs1)