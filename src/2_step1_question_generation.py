import pandas as pd
import numpy as np
from tqdm import tqdm
from utilities import openai_apis
from utilities import backward_func as bfunc

def CSL_work(path, input_name, output_name):
	# DATA STEP
	sample = pd.read_csv(path + input_name)
	#sample = sample[["dataset", "treatment", "treatment_name", "outcome", "outcome_name"]]
	sample = sample.replace({np.nan: ""})
	print(sample.head())

	## EXAMPLAR STEP
	inputs = pd.DataFrame({
	    "dataset": ["employee_data.csv", "job_data.csv", "weather_data.csv"],
	    "var": ["all_variables", "education, job satisfaction", "temperature, humidity, rainfall, local latitude, local altitude"],
	    "var_name": ["", "edu, job_satisfaction", "temp, humid, rain, lat, alt"]
	})

	examples1 = [
	    "In the dataset employee_data.csv, how many causal relationships can be detected?",
	    "Within the employee_data.csv dataset, how many causal links can be discerned?",
	    "How many causal connections can be identified in the employee_data.csv dataset?",
	    "Can you determine all causal connections present in the employee_data.csv dataset?",
	    "Within the employee_data.csv dataset, how many instances of one factor directly causing another can be observed?",
	    "Can we pinpoint the number of instances in the employee_data.csv dataset where one variable directly leads to another?",
	    "Is there a way to recognize every direct relationship in the employee_data.csv dataset where one variable influences another?",
	    "Is there a method to discover every direct influence present in the employee_data.csv dataset?"
	    ]

	examples2 = [
	    "Within the job_data.csv dataset, is there causal links between education (edu) and job satisfaction (job_satisfaction)?",
	    "Are there discernible causal links between education (edu) and job satisfaction (job_satisfaction) in the job_data.csv dataset?",
	    "Within the job_data.csv dataset, is there evidence of causal connections between education (edu) and job satisfaction (job_satisfaction)?",
	    "Does the job_data.csv data reveal any direct relationships between education (edu) and job satisfaction (job_satisfaction)?",
	    "Can we find any concrete connections between education (edu) and job satisfaction (job_satisfaction) in the job_data.csv dataset?",
	    "Do the records in job_data.csv show if education (edu) directly impacts job satisfaction (job_satisfaction)?",
	    "Is it evident in the job_data.csv dataset whether education (edu) plays a role in affecting job satisfaction (job_satisfaction)?",
	    "From the job_data.csv dataset, can we infer if education (edu) has a significant impact on job satisfaction (job_satisfaction)?"
	]

	examples3 = [
	    "What are causal links among temperature (temp), humidity (humid), rainfall (rain), local latitude (lat), and local altitude (alt) in dataset weather_data.csv?",
	    "Could you describe the causal relationships among temperature (temp), humidity (humid), rainfall (rain), local latitude (lat), and local altitude (alt) in the weather_data.csv dataset?",
	    "Within the weather_data.csv dataset, can you outline the causal connections among temperature (temp), humidity (humid), rainfall (rain), local latitude (lat), and local altitude (alt)?",
	    "In the weather_data.csv dataset, what causal connections can be identified among temperature (temp), humidity (humid), rainfall (rain), local latitude (lat), and local altitude (alt)?",
	    "Within the dataset weather_data.csv, what is the nature of the causal links among temperature (temp), humidity (humid), rainfall (rain), local latitude (lat), and local altitude (alt)?"
	]

	examplr_csl = [
	    (bfunc.generate_input_from_row("csl", inputs.iloc[0]), examples1),
	    (bfunc.generate_input_from_row("csl", inputs.iloc[1]), examples2),
	    (bfunc.generate_input_from_row("csl", inputs.iloc[2]), examples3)
	]
	#examplr_csl

	sample['prompt'] = sample.apply(lambda row: bfunc.generate_prompt("csl", row, examplr_csl), axis=1)
	print(sample.iloc[0]["prompt"])

	#XX = 0
	#print(bfunc.generate_input_from_row("csl", sample.iloc[XX]))
	#print(openai_apis.get_completion(sample['prompt'][XX], temp = .9))
	out = []
	for i in tqdm(range(len(sample))): #tqdm(range(20,30)): #
	  if i < len(out):
	    print("i =", i, "passed\n", sample['dataset'][i], "\n", out[i])
	    continue
	  ret = openai_apis.get_completion(sample['prompt'][i], temp = .9)
	  print("\n", "-" * 100, "\n", bfunc.generate_input_from_row("csl", sample.iloc[i]), "\n\n", ret)
	  out.append(ret)
	  #break
	sample['input'] = out

	## save the data somewhere
	sample.to_csv(path + output_name)

def ATE_work(path, input_name, output_name):
	## DATA STEP
	sample = pd.read_csv(path + input_name)
	sample = sample[["dataset", "treatment", "treatment_name", "outcome", "outcome_name"]]
	sample = sample.replace({np.nan: ""})
	#print(sample.head())

	## EXAMPLAR STEP
	inputs = pd.DataFrame({
	    "dataset": ["disaster_risk_reduction.csv", "patient_data.csv"],
	    "treatment": ["disaster risk reduction budget", "dosage of medication"],
	    "treatment_name": ["", "med_dosage"],
	    "outcome": ["community resilience index", "blood pressure"],
	    "outcome_name": ["c_resilience_index", "bp"]
	})

	examples1 = [
	    "By what margin does the disaster risk reduction budget influence the community resilience index (c_resilience_index) in the disaster_risk_reduction.csv dataset?",
	    "How significant are the changes to the community resilience index (c_resilience_index) with increases in the disaster risk reduction budget, based on the disaster_risk_reduction.csv dataset?",
	    "Using the disaster_risk_reduction.csv data, can we tell if a higher disaster risk reduction budget leads to a better score on the community resilience index (c_resilience_index)?",
	    "How impactful is the disaster risk reduction budget on the community resilience index (c_resilience_index) based on the findings from disaster_risk_reduction.csv?",
	    "In the disaster_risk_reduction.csv dataset, could you elucidate how fluctuations of the disaster risk reduction budget influence the community resilience index (c_resilience_index)?",
	    "Drawing from the disaster_risk_reduction.csv dataset, what inferences can be made about the influence of the disaster risk reduction budget on the community resilience index (c_resilience_index)?",
	    "To what extent does the disaster risk reduction budget impact the community resilience index (c_resilience_index) as shown in the disaster_risk_reduction.csv dataset?",
	    "From the insights in disaster_risk_reduction.csv, does an augmentation in the disaster risk reduction budget directly elevate the levels of the community resilience index (c_resilience_index)?",
	    "Given the information in disaster_risk_reduction.csv, how pivotal is the disaster risk reduction budget in shaping the trajectory of the community resilience index (c_resilience_index)?",
	    "Do fluctuations in the disaster risk reduction budget, as seen in disaster_risk_reduction.csv, have a pronounced effect on the readings of the community resilience index (c_resilience_index)?",
	    "How quantifiable is the impact of disaster risk reduction budget on the subsequent shifts in the community resilience index (c_resilience_index), as derived from the disaster_risk_reduction.csv dataset?",
	    "How substantial is the effect of fluctuations in the disaster risk reduction budget on the readings of the community resilience index (c_resilience_index) based on disaster_risk_reduction.csv data?",
	    "In the dataset disaster_risk_reduction.csv, how does the disaster risk reduction budget influence the community resilience index (c_resilience_index)?"
	    ]

	examples2 = [
	    "How does the dosage of medication (med_dosage) affect the patient's blood pressure (bp) in the patient_data.csv dataset?",
	    "In the patient_data.csv, does changing the dosage of medication (med_dosage) make a big change in blood pressure (bp)?",
	    "Using the patient_data.csv, if we give more or less dosage of medication (med_dosage), how much does blood pressure (bp) change?",
	    "From the patient_data.csv, can we see a clear change in blood pressure (bp) when the dosage of medication (med_dosage) is different?",
	    "Looking at the patient_data.csv, does a different dosage of medication (med_dosage) really change someone's blood pressure (bp) a lot?",
	    "Based on the patient_data.csv, how much does the dosage of medication (med_dosage) mess with someone's blood pressure (bp)?",
	    "From the patient_data.csv file, how big of a change in blood pressure (bp) do we see when the dosage of medication (med_dosage) goes up or down?"
	]

	examplr_ate = [
	    (bfunc.generate_input_from_row("ate", inputs.iloc[0]), examples1),
	    (bfunc.generate_input_from_row("ate", inputs.iloc[1]), examples2)
	]
	#print(examplr_ate)

	## GENERATE PROMPT
	sample['prompt'] = sample.apply(lambda row: bfunc.generate_prompt("ate", row, examplr_ate), axis=1)
	#print(sample.iloc[0]["prompt"])

	## COMPLETE PROMPT BY CHATGPT
	out = []
	for i in tqdm(range(len(sample))):
	  ret = openai_apis.get_completion(sample['prompt'][i], temp = .9)
	  print("-" * 100, "\n", ret)
	  out.append(ret)
	  #break
	sample['input'] = out

	## save the data somewhere
	sample.to_csv(path + output_name)

def HTE_work(path, input_name, output_name):
	## DATA STEP
	sample = pd.read_csv(path + input_name)
	sample1 = sample
	sample = sample.replace({np.nan: ""})
	sample.reset_index(inplace = True, drop = True)
	#print(sample.head())

	## EXAMPLAR STEP
	inputs = pd.DataFrame({
	    "dataset": ["disaster_risk_reduction.csv", "financial_inclusion.csv"],
	    "treatment": ["building code compliance rate", "financial inclusion index"],
	    "treatment_name": ["", "index"],
	    "outcome": ["risk assessment coverage", "microfinance coverage"],
	    "outcome_name": ["risk_assessment_coverage", "microfinance_coverage"],
	    "condition": ["disaster risk reduction budget", "financial literacy rate"],
	    "condition_name": ["budget", "literacy_rate"],
	    "condition_value": ["0.61", "0.41"],
	})

	examples1 = [
	    "What is the effect of the building code compliance rate on risk assessment coverage (risk_assessment_coverage) when the disaster risk reduction budget is 0.61 according to the data in disaster_risk_reduction.csv?",
	    "In the disaster_risk_reduction.csv data, how does the risk assessment coverage (risk_assessment_coverage) change when the building code compliance rate is considered and the disaster risk reduction budget is 0.61?",
	    "Based on the information in disaster_risk_reduction.csv, what impact does the building code compliance rate have on risk assessment coverage (risk_assessment_coverage) when the disaster risk reduction budget is 0.61?",
	    "According to the findings in disaster_risk_reduction.csv, when the disaster risk reduction budget is 0.61, how does the building code compliance rate affect the risk assessment coverage (risk_assessment_coverage)?",
	    "How does the risk assessment coverage (risk_assessment_coverage) change when the building code compliance rate is factored in, with a disaster risk reduction budget set at 0.61, as per the findings in disaster_risk_reduction.csv?",
	    "Based on the findings in disaster_risk_reduction.csv, how does the building code compliance rate impact risk assessment coverage (risk_assessment_coverage) when the disaster risk reduction budget is 0.61 (budget=0.61)?",
	    "Considering the data from disaster_risk_reduction.csv, what insights can be drawn regarding the impact from the building code compliance rate on risk assessment coverage (risk_assessment_coverage) with a disaster risk reduction budget of 0.61 (budget=0.61)?",
	    "Within the scope of disaster_risk_reduction.csv, how does the building code compliance rate impact risk assessment coverage (risk_assessment_coverage) when the disaster risk reduction budget is established at 0.61 (budget=0.61)?",
	    ]

	examples2 = [
	    "Examining financial_inclusion.csv, how does the microfinance coverage (microfinance_coverage) vary with changes in the financial inclusion index (index) when the financial literacy rate remains constant at 0.41 (literacy_rate=0.41)?",
	    "Within the dataset financial_inclusion.csv, what patterns emerge regarding microfinance coverage (microfinance_coverage) in relation to fluctuations in the financial inclusion index (index) while maintaining a fixed financial literacy rate of 0.41 (literacy_rate=0.41)?",
	    "In financial_inclusion.csv, how does the interplay between the financial inclusion index (index) and microfinance coverage (microfinance_coverage) unfold, especially when the financial literacy rate is consistent at 0.41 (literacy_rate=0.41)?",
	    "In the financial_inclusion.csv dataset, what trends are noticeable in microfinance coverage (microfinance_coverage) concerning changes in the financial inclusion index (index), keeping the financial literacy rate constant at 0.41 (literacy_rate=0.41)?",
	    "Looking at the financial_inclusion.csv data, are there specific trends in microfinance coverage (microfinance_coverage) concerning variations in the financial inclusion index (index), while maintaining a stable financial literacy rate of 0.41 (literacy_rate=0.41)?",
	    "Within financial_inclusion.csv, what kind of patterns are evident in microfinance coverage (microfinance_coverage) concerning changes in the financial inclusion index (index), given a constant financial literacy rate of 0.41 (literacy_rate=0.41)?",
	    "According to financial_inclusion.csv, are there any trends in microfinance coverage (microfinance_coverage) when examining fluctuations in the financial inclusion index (index), while keeping the financial literacy rate steady at 0.41 (literacy_rate=0.41)?"
	]

	examplr_hte = [
	    (bfunc.generate_input_from_row("hte", inputs.iloc[0]), examples1),
	    (bfunc.generate_input_from_row("hte", inputs.iloc[1]), examples2)
	]
	#print(examplr_hte)

	## GENERATE PROMPT
	sample['prompt'] = sample.apply(lambda row: bfunc.generate_prompt("hte", row, examplr_hte), axis=1)
	#print(sample.iloc[0]["prompt"])

	## COMPLETE PROMPT BY CHATGPT
	out = []
	for i in tqdm(range(len(sample))):
	  ret = openai_apis.get_completion(sample['prompt'][i], temp = .9)
	  print("-" * 100, "\n", ret)
	  out.append(ret)
	  #break
	sample['input'] = out

	## save the data somewhere
	sample.to_csv(path + output_name)

def MA_work(path, input_name, output_name):
	## DATA STEP
	sample = pd.read_csv(path + input_name)
	sample = sample[["dataset", "treatment", "treatment_name", "outcome", "outcome_name", "mediator", "mediator_name"]]
	sample = sample.replace({np.nan: ""})
	#print(sample.head())

	## EXAMPLAR STEP
	inputs = pd.DataFrame({
	    "dataset": ["smoke.csv", "arts_and_culture.csv"],
	    "treatment": ["smoking", "museum attendance"],
	    "treatment_name": ["smk", "visitor_count"],
	    "outcome": ["weight changes", "cultural diversity index"],
	    "outcome_name": ["", "CD"],
	    "mediator": ["appetite suppression", "cultural events per capita"],
	    "mediator_name": ["app_sup", ""]
	})

	examples1 = ["Is it evidenced by smoke.csv that the pathway from smoking (smk) to weight changes is substantially mediated by appetite suppression (app_sup)?",
	        "What is the causal effect of smoking (smk) on weight changes as depicted in the smoke.csv dataset, and how much of this effect is mediated through appetite suppression (app_sup)?",
	        "Using the smoke.csv data, can we quantify the causal effect size of smoking (smk) on weight changes, specifically assessing the mediation effect size attributed to appetite suppression (app_sup)?",
	        "Can we determine the effect size of the causal relationship between smoking (smk) and weight changes in the smoke.csv data, and the effect size of the mediating role of appetite suppression (app_sup)?",
	        "Using causal inference techniques on smoke.csv, how can we quantify the effect size of smoking (smk) on weight changes and disentangle the portion mediated by appetite suppression (app_sup)?",
	        "How can the smoke.csv data inform us about the specific effect sizes of smoking (smk) on weight changes, both through direct pathways and through indirect pathways mediated by appetite suppression (app_sup)?",
	        "Looking at the smoke.csv data, can we tell how much smoking (smk) influence a person's weight changes, and is a lot of that because of the appetite suppression (app_sup)?",
	        "From smoke.csv, how big of a change in weight changes does smoking (smk) cause, and is most of this change due to appetite suppression (app_sup)?",
	        "Does the smoke.csv dataset suggest that appetite suppression (app_sup) is a big reason why people's weight changes when smoking (smk)?",
	        "Looking through smoke.csv, can we see a pattern where smoking (smk)'s effect on weight changes is heavily mediated by appetite suppression (app_sup)?",
	        "Can we figure out from smoke.csv how much smoking (smk) typically impact a person's weight changes, and how much of this change happens because of appetite suppression (app_sup)?",
	        "From the insights provided by smoke.csv, can we quantify the mediating influence of appetite suppression (app_sup) on the smoking (smk)-->weight changes?"]

	examples2 = ["In the arts_and_culture.csv dataset, how significantly does cultural events per capita mediate the effect of museum attendance (visitor_count) on the cultural diversity index (CD)?",
	        "Can we measure the extent to which cultural events per capita serve as a mediator in the causal pathway from museum attendance (visitor_count) to cultural diversity index (CD), as represented in arts_and_culture.csv?",
	        "To what extent does the mediator, cultural events per capita, modify or influence the impact of museum attendance (visitor_count) on the cultural diversity index (CD) in the arts_and_culture.csv dataset?",
	        "How substantial is the mediating effect of cultural events per capita on the relationship between museum attendance (visitor_count) and the cultural diversity index (CD) within the context of the arts_and_culture.csv data?",
	        "In the arts_and_culture.csv, can we figure out how big a role cultural events per capita plays in linking museum attendance (visitor_count) to cultural diversity index (CD)?",
	        "When looking at the arts_and_culture.csv, how much does cultural events per capita get in the middle of the effect of museum attendance (visitor_count) on cultural diversity index (CD)?",
	        "Can we pinpoint how forcefully cultural events per capita intervenes in the connection between museum attendance (visitor_count) and cultural diversity index (CD) within the arts_and_culture.csv dataset?",
	        "Does cultural events per capita serve as a key stepping stone in the causal relationship from museum attendance (visitor_count) to cultural diversity index (CD) in the context of the arts_and_culture.csv?",
	        "Regarding the arts_and_culture.csv, how central is the role of cultural events per capita in the direct causal flow from museum attendance (visitor_count) to impacting cultural diversity index (CD)?",
	        "In the arts_and_culture.csv dataset, how much of the effect of museum attendance (visitor_count) on the cultural diversity index (CD) is directly from the attendance itself, and how much is indirectly through cultural events per capita?",
	        "Regarding the data from arts_and_culture.csv, can we distinguish the direct impact of museum attendance (visitor_count) on cultural diversity index (CD) from the indirect impact that operates through cultural events per capita?",
	        "In the arts_and_culture.csv, can we split up how much museum attendance (visitor_count) directly affects cultural diversity index (CD) from the part that's because of cultural events per capita?"     ]

	examplr_ma = [
	    (bfunc.generate_input_from_row("ma", inputs.iloc[0]), examples1),
	    (bfunc.generate_input_from_row("ma", inputs.iloc[1]), examples2)
	]
	print(examplr_ma)

	## GENERATE PROMPT
	sample['prompt'] = sample.apply(lambda row: bfunc.generate_prompt("ma", row, examplr_ma), axis=1)
	print(sample.iloc[0]["prompt"])

	## COMPLETE PROMPT BY CHATGPT
	out = []
	for i in tqdm(range(len(sample))):
	  ret = openai_apis.get_completion(sample['prompt'][i], temp = .9)
	  print("-" * 100, "\n", ret)
	  out.append(ret)
	  #break
	sample['input'] = out

	## save the data somewhere
	sample.to_csv(path + output_name)

def CPL_work(path, input_name, output_name):
	## DATA STEP
	sample = pd.read_csv(path + input_name)
	sample = sample.replace({np.nan: ""})
	#print(sample.head())

	inputs = pd.DataFrame({
	    "dataset": ["diabetes.csv", "DRR.csv"],
	    "treatment": ["insulin dose", "disaster preparedness plan"],
	    "treatment_name": ["Dose", ""],
	    "outcome": ["blood glucose condition", "building code compliance rate"],
	    "outcome_name": ["glucose", "BCC_rate"],
	    "condition": ["exercise frequency", "community resilience"],
	    "condition_name": ["exercise", ""],
	    "condition_value": ["2", "0.07"],
	})

	examples1 = ["If my exercise frequency is consistently at a value of 2 (exercise=2), how should I adjust my insulin dose (Dose) using the data from the diabetes.csv dataset to best manage my blood glucose condition (glucose)?",
	        "Based on the diabetes.csv dataset, what insulin dose (Dose) strategy is recommended for someone who exercises with a frequency value of 2 (exercise=2) to achieve optimal blood glucose condition (glucose) levels?",
	        "Using insights from the diabetes.csv file, what action should a person with a consistent exercise frequency of 2 (exercise=2) take in terms of their insulin dose (Dose) to improve their blood glucose condition (glucose)?",
	        "Looking at the diabetes.csv data, what changes to insulin dose (Dose) might be advised for an individual with an exercise frequency of 2 (exercise=2) to effectively manage their blood glucose condition (glucose) levels?",
	        "Could you elaborate on how the diabetes.csv dataset informs insulin dose (Dose) decisions for patients who consistently exercise at a frequency value of 2 (exercise=2), aiming to optimize their blood glucose condition (glucose)?",
	        "In light of the patient's exercise frequency being at a value of 2 (exercise=2), how does the diabetes.csv data guide us in fine-tuning their insulin dose (Dose) for better blood glucose condition (glucose)?",
	        "Based on the patient's data in the diabetes.csv file and their exercise frequency of 2 (exercise=2), can you suggest an appropriate insulin dose (Dose) strategy to optimize their blood glucose condition (glucose)?",
	        "Considering the patient's consistent exercise frequency of 2 (exercise=2), what specific insights from the diabetes.csv dataset can guide me in determining the right insulin dose (Dose) for optimizing their monitored blood glucose condition (glucose)?",
	        "Hey Doc, with the diabetes.csv data in hand, could you provide clear guidance on managing my insulin dose (Dose) given my exercise frequency is at 2 (exercise=2)? I want to make sure my blood glucose condition (glucose) is in the best state possible.",
	        "Could you use the diabetes.csv data to guide me on adjusting my insulin dose (Dose), considering my exercise frequency is 2 (exercise=2), for optimal blood glucose condition (glucose)?",
	        "So, looking at the diabetes.csv, how should I tweak my insulin dose (Dose) if I'm usually exercising at a level of 2 (exercise=2)? I want to keep my blood glucose condition (glucose) good.",
	        "Can you help me out with what to do with my insulin dose (Dose), based on the diabetes.csv you got, especially since I'm exercising at a level 2 (exercise=2)? I need to keep my blood glucose condition (glucose) in check."]

	# add a "natrual example" : maybe from recommender system / game?
	examples2 = ["Given that our community resilience is currently at 0.07, does the DRR.csv data suggest that adopting the disaster preparedness plan would significantly boost our building code compliance rate (BCC_rate)?",
	        "In light of our low community resilience value of 0.07, does the DRR.csv data provide evidence that following the disaster preparedness plan will be an effective strategy for enhancing building code compliance rate (BCC_rate)?",
	        "Considering our community resilience is just 0.07, can the DRR.csv dataset guide us on whether implementing the disaster preparedness plan is a necessary action to improve our building code compliance rate (BCC_rate)?",
	        "With a community resilience score of 0.07, does the information in the DRR.csv suggest that prioritizing the disaster preparedness plan will effectively optimize our building code compliance rate (BCC_rate)?",
	        "In the context of our community resilience score standing at 0.07, does the empirical evidence provided in the DRR.csv dataset advocate for the implementation of the disaster preparedness plan as a means to optimize our building code compliance rate (BCC_rate)?",
	        "With our community resilience measured at 0.07, does the DRR.csv dataset offer sufficient insights to conclude that prioritizing the disaster preparedness plan would be a strategic approach to significantly improve our building code compliance rate (BCC_rate)?",
	        "If we look at our community resilience sitting at just 0.07, does diving into the DRR.csv give us a thumbs-up to roll out that disaster preparedness plan, especially if we're aiming to ramp up the building code compliance rate (BCC_rate)?",
	        "Upon examining the DRR.csv dataset, do we have a clear indication to implement the disaster preparedness plan, particularly given that our community resilience is quantified at just 0.07 and our goal is to enhance the building code compliance rate (BCC_rate)?",
	        "Could you recommend, based on what the DRR.csv shows and factoring in our community resilience of 0.07, whether implementing the disaster preparedness plan would be beneficial for enhancing the building code compliance rate (BCC_rate)?",
	        "When we take a good look at what the DRR.csv's got and think about our community resilience being only 0.07, would you say giving the green light to that disaster preparedness plan is the way to go for bumping up our building code compliance rate (BCC_rate)?",
	        "Given the information from the DRR.csv and our current community resilience score of 0.07, would initiating the disaster preparedness plan, in your view, lead us toward a significant improvement in our building code compliance rate (BCC_rate)?",
	        "From a strategic standpoint, and upon examining the DRR.csv dataset in conjunction with our community resilience score of 0.07, do you suggest that prioritizing the disaster preparedness plan could be instrumental in driving substantial improvements in our building code compliance rate (BCC_rate)?"]

	examplr_cpl_s = [
	    (bfunc.generate_input_from_row("s_cpl", inputs.iloc[0]), examples1),
	    (bfunc.generate_input_from_row("s_cpl", inputs.iloc[1]), examples2)
	]
	#print(examplr_cpl_s)

	## GENERATE PROMPT
	sample['prompt'] = sample.apply(lambda row: bfunc.generate_prompt("s_cpl", row, examplr_cpl_s), axis=1)
	#print(sample.iloc[0]["prompt"])

	## COMPLETE PROMPT BY CHATGPT
	out = []
	for i in tqdm(range(len(sample))):
	# for i in tqdm(range(2)): # try a quick run
	  ret = openai_apis.get_completion(sample['prompt'][i], temp = .9)
	  print("-" * 100, "\n", ret)
	  out.append(ret)
	  # break
	sample['input'] = out

	## save the data somewhere
	## need to save the output out somewhere
	sample['input'] = out
	sample.to_csv(path + output_name)

def JSON_work(path):
	#CSL
	sample = pd.read_csv(path + "p30_CSL_v2.csv")
	#sample = sample[["dataset", "treatment", "treatment_name", "outcome", "outcome_name"]]
	sample = sample.replace({np.nan: ""})
	sample.head()
	sample['output'] = sample.apply(lambda row: bfunc.variable2output('CSL',row['dataset'], nodes = row['var'], nodes_name = row['var_name']), axis=1)
	sample.to_csv(path + 'CSL_JSON_output.csv')

	#ATE
	sample = pd.read_csv(path + "p30_ATE_v2.csv")
	sample = sample.replace({np.nan: ""})
	sample.head()
	sample['output'] = sample.apply(lambda row: bfunc.variable2output('ATE',row['dataset'], treatment = row['treatment'],
	                                trt_name = row['treatment_name'], response = row['outcome'],
	                               res_name = row['outcome_name']), axis=1)
	sample.to_csv(path + 'ATE_JSON_output.csv')

	#HTE
	sample = pd.read_csv(path + "p30_HTE_v2.csv")
	sample = sample.replace({np.nan: ""})
	sample.head()
	sample['output'] = sample.apply(lambda row: bfunc.variable2output('HTE',row['dataset'], treatment = row['treatment'],
	                                trt_name = row['treatment_name'], response = row['outcome'],
	                                res_name = row['outcome_name'], condition = row['condition'],
	                                condition_name = row['condition_name'], condition_value = row['condition_value']), axis=1)
	sample.to_csv(path + 'HTE_JSON_output.csv')

	#MA
	sample = pd.read_csv(path + "p30_MA_v2.csv")
	sample = sample.replace({np.nan: ""})
	sample.head()
	sample['output'] = sample.apply(lambda row: bfunc.variable2output('MA',row['dataset'], treatment = row['treatment'],
	                                trt_name = row['treatment_name'], response = row['outcome'],
	                                res_name = row['outcome_name'], mediator = row['mediator'],
	                                med_name = row['mediator_name']), axis=1)
	sample.to_csv(path + 'MA_JSON_output.csv')

	#CPL
	sample = pd.read_csv(path + "p30_CPL_v2.csv")
	sample = sample.replace({np.nan: ""})
	sample.head()
	sample['output'] = sample.apply(lambda row: bfunc.variable2output('CPL',row['dataset'], treatment = row['treatment'],
	                                trt_name = row['treatment_name'], response = row['outcome'],
	                                res_name = row['outcome_name'], condition = row['condition'],
	                                condition_name = row['condition_name'], condition_value = row['condition_value']), axis=1)
	sample.to_csv(path + 'CPL_JSON_output.csv')

def split_outputs(path):
	file = [
    "p30_CSL_v2.csv",
    "p30_ATE_v2.csv",
    "p30_HTE_v2.csv",
    "p30_MA_v2.csv",
    "p30_CPL_v2.csv"
	]

	outs = [
	    "CSL_JSON_output.csv",
	    "ATE_JSON_output.csv",
	    "HTE_JSON_output.csv",
	    "MA_JSON_output.csv",
	    "CPL_JSON_output.csv"
	]

	for f, o  in zip(file, outs):
	  data = pd.read_csv(path + f)
	  data_j = pd.read_csv(path + o)
	  data_k = data.join(data_j[["output"]])

	  data_p = bfunc.split_df_input(data_k)
	  data_p = data_p[[x for x in data_p.columns if not x[:7] == "Unnamed"]]
	  print(len(data_p), data_p.columns)
	  data_p.to_csv(path + f[:-4] + "_check.csv")
	  #break

	data_p

if __name__ == '__main__':
	## path
	path = "../data/run_files/"

	## CSL
	#CSL_work(path = path, input_name = "CSL_30_v2.csv", output_name = "p30_CSL_v2.csv")

	## ATE
	#ATE_work(path = path, input_name = "ATE_30_v2.csv", output_name = "p30_ATE_v2.csv")

	## HTE
	#HTE_work(path = path, input_name = "HTE_30_v2.csv", output_name = "p30_HTE_v2.csv")

	## MA
	#MA_work(path = path, input_name = "MA_30_v2.csv", output_name = "p30_MA_v2.csv")
	
	## CPL
	#CPL_work(path = path, input_name = "CPL_30_v2.csv", output_name = "p30_CPL_v2.csv")

	## Create JSON output
	JSON_work(path = path)

	## Split into long format
	split_outputs(path)
