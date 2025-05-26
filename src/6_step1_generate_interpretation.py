import pandas as pd
import numpy as np
import json

import os
import pickle
import random
import shutil
from retry import retry
from tqdm import tqdm
from io import StringIO
from collections import OrderedDict

import itertools
from utilities import openai_apis

action_classes = ['0', '1', '2', 'A', 'B', 'C', 'I', 'II', 'III']
def generate_templated_outcome(JSON_output, var = None):
  JSON_output = eval(JSON_output)
  if JSON_output['causal_problem'][0] == "CSL":
    if JSON_output['nodes'][0] != "all_variables":
      nodes = JSON_output['nodes']
    else:
      nodes = [x.strip() for x in var.split(',')]
    pairs = list(itertools.combinations(nodes, 2))
    if len(pairs)>1:
      random.shuffle(pairs)
      result = "There are " + str(max(2, len(pairs)//3)) + " pairs of significant causal relationships. "
      for j in range(max(2, len(pairs)//3)):
        nodes = list(pairs[j])
        random.shuffle(nodes)
        result += "The " + nodes[0] + " would causally influence the " + nodes[1] + "."
        if j < max(2, len(pairs)//3)-1:
          result += " "
    else:
      nodes = list(pairs[0])
      random.shuffle(nodes)
      result = "The " + nodes[0] + " would causally influence the " + nodes[1] + "."
  elif JSON_output['causal_problem'][0] == "CPL":
    result = "The best action of the "+JSON_output['treatment'][0]+' is '+JSON_output['treatment'][0]+' = '+random.choice(action_classes)+'.'
  elif JSON_output['causal_problem'][1] == "ATE":
    result = "The average treatment effect of setting "+JSON_output['treatment'][0]+' as 1 on the '+JSON_output['response'][0] +' is '+ str(round(random.uniform(-10.0, 10.0),2)) +'.'
  elif JSON_output['causal_problem'][1] == "HTE":
    result = "The heterogeneous treatment effect of setting "+JSON_output['treatment'][0]+' as 1 on the '+JSON_output['response'][0] +' is '+ str(round(random.uniform(-10.0, 10.0),2)) +' for those having '+ JSON_output['condition'][0][0] + ' = ' + str(JSON_output['condition'][0][1]) + '.'
  elif JSON_output['causal_problem'][1] == "MA":
    DE = round(random.uniform(-10.0, 10.0),2)
    IE = round(random.uniform(-10.0, 10.0),2)
    TE = round(DE+IE,2)

    result = "The overall impact of the "+JSON_output['treatment'][0]+' on the '+JSON_output['response'][0] +' is '+ str(TE) +'. '
    result += "This comprises a direct effect of "+ str(DE) + " from the " + JSON_output['treatment'][0]+' to the '+JSON_output['response'][0]
    result += 'and an indirect effect of '+ str(IE) +', mediated by the ' + JSON_output['mediator'][0]+ '.'
  return result

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

if __name__ == '__main__':
	## input files
	path = "../data/run_files/p30_"
	ext = "_V2_sanity_.csv"
	ATE = pd.read_csv(path + 'ATE' + ext)
	ATE = ATE[['checked_input', 'output']]
	HTE = pd.read_csv(path + 'HTE' + ext)
	HTE = HTE[['checked_input', 'output']]
	MA = pd.read_csv(path + 'MA' + ext)
	MA = MA[['checked_input', 'output']]
	CSL = pd.read_csv(path + 'CSL' + ext)
	CSL = CSL[['checked_input', 'output']]
	CPL = pd.read_csv(path + 'CPL' + ext)
	CPL = CPL[['checked_input', 'output']]

	## override var if "all_variables" for CSL
	# read entity
	var_rep = pd.read_csv(path[:-4] + "entity_w_names.csv")
	datasets, var = [], []
	for idx, row in var_rep.iterrows():
		data_name = row["field"].strip() + ".csv"
		data_name = "_".join(data_name.split())
		if data_name not in datasets:
			datasets.append(data_name)
			var.append(row["variable"])
		else:
			pos = datasets.index(data_name)
			var[pos] += ", " + row["variable"]
	var_rep = pd.DataFrame({"datasets": datasets, "var": var})
	# process vars
	datasets = []
	for idx, row in CSL.iterrows():
		JSON_output = eval(row["output"])
		datasets.append(JSON_output["dataset"][0])
	CSL["datasets"] = datasets
	CSL.datasets = CSL.datasets.astype(str)
	var_rep.datasets = var_rep.datasets.astype(str)
	CSL = CSL.merge(var_rep, on = "datasets")

	## apply templated outcome
	HTE['templated_outcome'] = HTE.apply(lambda row: generate_templated_outcome(row['output']), axis=1)
	ATE['templated_outcome'] = ATE.apply(lambda row: generate_templated_outcome(row['output']), axis=1)
	MA['templated_outcome'] = MA.apply(lambda row: generate_templated_outcome(row['output']), axis=1)
	CPL['templated_outcome'] = CPL.apply(lambda row: generate_templated_outcome(row['output']), axis=1)
	CSL['templated_outcome'] = CSL.apply(lambda row: generate_templated_outcome(row['output'], row['var']), axis=1)

	HTE = HTE.sample(frac=1, random_state=42)  # random_state is optional
	HTE = HTE.reset_index(drop=True)

	ATE = ATE.sample(frac=1, random_state=42)  # random_state is optional
	ATE = ATE.reset_index(drop=True)

	MA = MA.sample(frac=1, random_state=42)  # random_state is optional
	MA = MA.reset_index(drop=True)

	CPL = CPL.sample(frac=1, random_state=42)  # random_state is optional
	CPL = CPL.reset_index(drop=True)

	CSL = CSL.sample(frac=1, random_state=42)  # random_state is optional
	CSL = CSL.reset_index(drop=True)

	ext1 = "_V2_outcome.csv"
	HTE.to_csv(path + 'HTE' + ext1)
	ATE.to_csv(path + 'ATE' + ext1)
	MA.to_csv(path + 'MA' + ext1)
	CPL.to_csv(path + 'CPL' + ext1)
	CSL.to_csv(path + 'CSL' + ext1)

	## generate interpretation by chatgpt
	query_example = [
	    "Does the disaster_risk_reduction.csv dataset provide evidence of a direct link between building code compliance rate and the effectiveness of disaster preparedness campaigns?",
	    "In the disaster_risk_reduction.csv dataset, how does the coverage of risk assessment (coverage_risk) influence the rate of building code compliance (compliance_rate)?",
	    "How does the community resilience index in the disaster_risk_reduction.csv dataset affect the building code compliance rate when the disaster preparedness campaigns are set at 0.07?",
	    "How can the disaster_risk_reduction.csv dataset help us estimate the magnitude of the impact of building code compliance rate (compliance_rate) on the effectiveness of disaster preparedness campaigns, while considering the role of the disaster risk reduction budget as a mediator?",
	    "Can you provide guidance on the most effective action to take in disaster preparedness campaigns (disaster_campaigns), considering the disaster_risk_reduction.csv dataset, and specifically targeting community resilience index (resilience_index) at a level of 0.07, in order to maximize building code compliance rate (building_code)?",
	]

	JSON_example = [
	      "{'causal_problem': ['CSL', None], 'dataset': ['disaster_risk_reduction.csv'], 'nodes': ['building_code_compliance_rate', 'disaster_preparedness_campaigns']}",
	      "{'causal_problem': ['CEL', 'ATE'], 'dataset': ['disaster_risk_reduction.csv'], 'treatment': ['coverage_risk'], 'response': ['compliance_rate']}",
	      "{'causal_problem': ['CEL', 'HTE'], 'dataset': ['disaster_risk_reduction.csv'], 'treatment': ['community_resilience_index'], 'response': ['building_code_compliance_rate'], 'condition': [('campaigns', '0.07')]}",
	      "{'causal_problem': ['CEL', 'MA'], 'dataset': ['disaster_risk_reduction.csv'], 'treatment': ['compliance_rate'], 'response': ['disaster_preparedness_campaigns'], 'mediator': ['disaster_risk_reduction_budget']}",
	      "{'causal_problem': ['CPL', None], 'dataset': ['disaster_risk_reduction.csv'], 'treatment': ['disaster_campaigns'], 'response': ['building_code'], 'condition': [('resilience_index', '0.07')]}",
	  ]


	## structured output
	output_example = [
	    "building_code_compliance_rate leads to disaster_preparedness_campaigns",
	    "the average treatment effect of coverage_risk on the compliance_rate is 0.02",
	    "the heterogeneous treatment effect of the community_resilience_index on the building_code_compliance_rate is 0.4 for those having campiagns = 0.07",
	    "the direct effect of compliance_rate on the disaster_preparedness_campaigns is 0.1, and the indirect effect of compliance_rate on the disaster_preparedness_campaigns that mediated by the disadter_risk_reduction_budget is 0.3",
	    "The best action on disaster_campaigns is class 1",
	]

	output_example_beta = [
	    "building_code_compliance_rate leads to disaster_preparedness_campaigns",
	    "the average treatment effect of setting coverage_risk = 1 on the compliance_rate is 0.02",
	    "the heterogeneous treatment effect of setting community_resilience_index = 1 on the building_code_compliance_rate is 0.4 for those having campiagns = 0.07",
	    "the direct effect of compliance_rate on the disaster_preparedness_campaigns is 0.1, and the indirect effect of compliance_rate on the disaster_preparedness_campaigns that mediated by the disadter_risk_reduction_budget is 0.3",
	    "The best action on disaster_campaigns is class 1",
	]


	s = 0
	size = 20
	test = pd.concat([ATE.iloc[s*size:(s+1)*size], HTE.iloc[s*size:(s+1)*size],
	                    MA.iloc[s*size:(s+1)*size], CSL.iloc[s*size:(s+1)*size], CPL.iloc[s*size:(s+1)*size]])
	test = test.reset_index(drop=True)
	test['prompt4interpret'] = test.apply(lambda row: generate_interpretation(row['checked_input'], row['output'], row['templated_outcome']), axis=1)
	test.to_csv(path + "inter_ready.csv")
	prompt4interpret, model, interp, ind = [], [], [], []
	for mod in ["gpt-3.5-turbo", "gpt-4-1106-preview"]:
	  	for i in tqdm(range(len(test))):
		  # for i in tqdm(range(2)): # try a quick run
		    ret = openai_apis.get_completion(
		    	test['prompt4interpret'][i], 
		    	model = mod,
		    	temp = .9)
		    print("-" * 100, "\n", ret)
		    prompt4interpret.append(test['prompt4interpret'][i])
		    model.append(mod)
		    interp.append(ret)
		    ind.append(i)
	interp_gpts = pd.DataFrame({
		"prompt4interpret": prompt4interpret,
		"model": model,
		"index": ind,
		"interpret": interp
		})
	interp_gpts.to_csv(path + "gpts_inter.csv")
