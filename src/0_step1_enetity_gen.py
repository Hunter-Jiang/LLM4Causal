from utilities import entity_generation

import pandas as pd 
import numpy as np 
from tqdm import tqdm

if __name__ == '__main__':
	# local run indicator
	run = True

	# generate topics and variables
	df_out = entity_generation.generate_entity_once()
	df_out.to_csv("../data/run_files/entity_raw.csv")

	# if we wants to held out a part of it
	held = True
	if held:
		df_train = pd.read_csv("../data/replicates/variable_name.csv")
		exists = list(df_train["field"])
		exists = [x.lower().strip() for x in exists]

		can_use = [x for x in df_out["Input"] if x.lower().strip() not in exists]
		print(can_use, "\n", len(can_use))
		if len(can_use) < 10:
			run = False

	# label variable names and their categories
	if run:
		field, variable, name, category = [], [], [], []
		for idx, row in tqdm(df_out.iterrows()):
		    if row["Input"].lower().strip() in exists:
		        continue
		    f = row["Input"]
		    vs = [x.strip() for x in row["Output"].split(",")]
		    for v in vs:
		        n, c = entity_generation.generate_name_for_x(v)
		        field.append(f)
		        variable.append(v)
		        name.append(n)
		        category.append(c)
		    #processed += 1

		data_name = pd.DataFrame({
		    "field": field,
		    "variable": variable,
		    "name": name,
		    "category": category
		})
		print(data_name.head())
		data_name.to_csv("../data/run_files/entity_w_names.csv")