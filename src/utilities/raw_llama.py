import pandas as pd
import numpy as np 
import matplotlib
import pickle
import os

	 
if __name__ in "__main__":
	data = pd.read_csv("p30_inter_ready.csv")
	print(data.head())

	with open("llama_30p_raw.pkl", "rb") as handle:
		llama = pickle.load(handle)

	data["llama2"] = llama
	data.to_csv("llama_interpret.csv", index= False)
