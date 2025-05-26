#inter_30p_s3.pkl

import pandas as pd 
import numpy as np 
import pickle

if __name__ in "__main__":
	## gpts
	path1 = "../data/run_files/"
	data = pd.read_csv(path1 + "p30_gpts_inter.csv")


	## ours
	path = "../data/replicates/"
	files = ["inter_30p_mixed.pkl", "inter_30p_s3.pkl"]

	## create dataframe
	interpret = []
	for file in files:
		with open(path + file, "rb") as handle:
			tmp = pickle.load(handle)
		interpret += tmp

	interpret = [x.replace("</s>", "") for x in interpret]
	data1 = data.copy()
	data1["model"] = ["mixed"] * 100 + ["s3_only"] * 100
	data1["interpret"] = interpret

	#quit()

	## random shuffle
	data_all = pd.concat([data, data1])
	data_all["random_int"] = np.random.randint(0,100, size = 400)
	data_all = data_all.sort_values(by = ["index", "random_int"])
	data_all.to_csv(path1 + "p30_merge_together.csv")

	## split by index
	data_all[data_all["index"] < 50][[
		"index", 
		"prompt4interpret", 
		"interpret"
		]].to_csv(path1 + "check_part1.csv", index=False)
	data_all[data_all["index"] >= 50][[
		"index", 
		"prompt4interpret", 
		"interpret"
		]].to_csv(path1 + "check_part2.csv", index=False)
