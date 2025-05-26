import pandas as pd 
import numpy as np
import itertools
import pickle
import random
import os

action_classes = ['0', '1', '2', 'A', 'B', 'C', 'I', 'II', 'III']
def generate_function_outcome(JSON_output, function_output, data):
	JSON_output = eval(JSON_output)
	if JSON_output['causal_problem'][0] == "CSL":
		pairs = mat2pairs(data, function_output)
		#print(pairs)
		#print(function_output)
		if len(pairs)>1:
			random.shuffle(pairs)
			result = "There are " + str(max(2, len(pairs)//3)) +" pairs of significant causal relationships. "
			for j in range(max(2, len(pairs)//3)):
				nodes = list(pairs[j])
				#random.shuffle(nodes)
				result += "The " + nodes[0] + " would causally influence the " + nodes[1] + "."
				if j < max(2, len(pairs)//3)-1:
					result += " "
		elif len(pairs) == 1:
			nodes = list(pairs[0])
			result = "The " + nodes[0] + " would causally influence the " + nodes[1] + "."
		else:
			result = "No causal link identified."
	elif JSON_output['causal_problem'][0] == "CPL":
		result = "The best action of the "+JSON_output['treatment'][0]
		result += ' is '+JSON_output['treatment'][0]+' = '
		result += action_classes[function_output[0]]+'.'
	elif JSON_output['causal_problem'][1] == "ATE":
		result = "The average treatment effect of setting "
		result += JSON_output['treatment'][0]+' as 1 on the '
		result += JSON_output['response'][0] +' is '
		result += str(round(function_output,2)) +'.'
	elif JSON_output['causal_problem'][1] == "HTE":
		result = "The heterogeneous treatment effect of setting "
		result += JSON_output['treatment'][0] +' as 1 on the '
		result += JSON_output['response'][0] +' is '
		result += str(round(function_output,2)) +' for those having '+ JSON_output['condition'][0][0] + ' = ' + str(JSON_output['condition'][0][1]) + '.'
	elif JSON_output['causal_problem'][1] == "MA":
		DE = round(function_output[0],2)
		IE = round(function_output[1],2)
		TE = round(function_output[2],2)
		result = "The overall impact of the "+JSON_output['treatment'][0]+' on the '+JSON_output['response'][0] +' is '+ str(TE) +'. '
		result += "This comprises a direct effect of "+ str(DE) + " from the " + JSON_output['treatment'][0]+' to the '+JSON_output['response'][0]
		result += 'and an indirect effect of '+ str(IE) +', mediated by the ' + JSON_output['mediator'][0]+ '.'
	return result

def mat2pairs(data, matrix):
	columns = [x for x in list(data.columns) if x.find("Unnamed:") == -1]
	pairs = []
	for posx, row in enumerate(matrix):
		for posy, col in enumerate(row):
			if abs(col) > 0.05:
				pairs.append((col, [columns[posx], columns[posy]]))
	if len(pairs) > 0:
		pairs.sort(reverse = True, key = lambda x: x[0])
		pairs = [x[1] for x in pairs]
	return pairs

def generate_problem(JSON_in):
  JSON_eval = eval(JSON_in)
  match JSON_eval["causal_problem"]:
    case ['CSL', None]:
      problem, method = "causal structure learning", "PC algorithm" # or pc algrithm? sounds strange if it is the only method here
    case ['CEL', 'ATE']:
      problem, method = "average treatment effect", "doubly robust estimator"
    case ['CEL', 'HTE']:
      problem, method = "heterogeneous treatment effect", "S-learner"
    case ['CEL', 'MA']:
      problem, method = "mediation analysis", "doubly robust estimator"
    case ['CPL', None]:
      problem, method = "causal policy learning", "Q learning"
    case _:
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

if __name__ in "__main__":
	# get prediction
	df = pd.read_csv("evaluate_p30.csv")[["input", "output"]]
	with open("step1_mixed_0_run0.pkl", "rb") as handle:
		pkl = pickle.load(handle)
		pkl = [x[:-4] for x in pkl]
	df["predict"] = pkl
	# get function outputs
	result_list = []
	for idx, row in df.iterrows():
		path = "datafiles/" + str(idx)
		#print(idx, path)
		try:
			with open(path + "/output.pkl", "rb") as handle:
				output = pickle.load(handle)
			data = pd.read_csv(path + "/" + [x for x in os.listdir(path) if x[-4:] == ".csv"][0])
			#print(output)
			res = generate_function_outcome(row["predict"], output, data)
			result_list.append(res)
		except:
			result_list.append("Failed to run that function.")
		#print(res)
		#print("-" * 50)

	df['templated_outcome'] = result_list
	df['prompt4interpret'] = df.apply(lambda row: generate_interpretation(row['input'], row['output'], row['templated_outcome']), axis=1)
	print(df.head())
	df1 = df[["prompt4interpret"]]
	df1.columns = ["input"]
	df1["output"] = ""
	df1["type"] = "step3"
	df.to_csv("interpret_save_v2.csv")
	df1.to_csv("interpret_ready_go_v2.csv")