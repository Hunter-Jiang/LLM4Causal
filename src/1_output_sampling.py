import pandas as pd
import numpy as np

import random

def dataset_format(field):
    field=field.lower()
    output="_".join(field.split())+".csv"
    return(output)

def random_mask(df, col_list, probability):
	for idx in df.index:
		if random.random() < probability:
			for col in col_list:
				df.at[idx, col] = None

def create_condition(variable,variable_name):
    if variable in categorical_list:
        variable_class=np.random.choice(class_list, 1)[0]
        variable_condition=variable+" is of class "+str(variable_class)
        variable_name_condition=variable_name+"="+str(variable_class)
    else:
        variable_class=np.random.choice(proportion_list, 1)[0]
        variable_condition=" has the "+variable+" of value "+str(variable_class)
        variable_name_condition=variable_name+"="+str(variable_class)
    return(variable_condition,variable_name_condition)

def create_condition_2(variable,variable_name):
    if variable in categorical_list:
        variable_class = np.random.choice(class_list, 1)[0]
        variable_class = 'class '+str(variable_class)
        #variable_condition=variable+" is of class "+str(variable_class)
        #variable_name_condition=variable_name+"="+str(variable_class)
    else:
        variable_class = str(np.random.choice(proportion_list, 1)[0])
        #variable_condition=" has the "+variable+" of value "+str(variable_class)
        #variable_name_condition=variable_name+"="+str(variable_class)
    return str(variable_class)

def create_condition_1(variable,variable_name):
  if variable in categorical_list:
      variable_class=np.random.choice(class_list, 1)[0]
      variable_condition=variable+","+str(variable_class)
      variable_name_condition=variable_name+"="+str(variable_class)
  else:
      variable_class=np.random.choice(proportion_list, 1)[0]
      variable_condition= variable+","+str(variable_class)
      variable_name_condition=variable_name+"="+str(variable_class)
  return(variable_condition,variable_name_condition)

def gen_var_name(subset_field,variable):
    subset_var=subset_field[subset_field['variable']==variable]
    var_name_list=subset_var["name"].item().split(",")
    var_name=random.sample(var_name_list,1)[0]
    return(var_name)

if __name__ == '__main__':
	path = "../data/run_files/"
	# reading enttities
	df_variable_name=pd.read_csv(path + "entity_w_names.csv")
	#print(df_variable_name.head())

	# creating dataset names
	df_variable_name["variable"]=df_variable_name["variable"].str.lower()
	df_variable_name["field"]=df_variable_name["field"].apply(lambda x: dataset_format(x))
	df_variable_name["name"]=df_variable_name["name"].apply(lambda x: x.replace(" ", ""))
	#print(df_variable_name.head())

	# random sample
	field_list=df_variable_name["field"].unique()

	np.random.seed(12345)
	n_sample = 6
	random_field_list=np.random.choice(25, n_sample, replace=True)
	#print(random_field_list)

	## CSL
	data_all=[]
	for i in random_field_list:
	    data_entry=[]
	    field_name=field_list[i]
	    data_entry.append(field_name)##generate field_name
	    subset_field=df_variable_name[df_variable_name['field']==field_name]
	    variable_list=subset_field["variable"].unique()
	    num_variable=np.random.choice(range(2,len(variable_list)+1),1, replace=False)## number of variable requested
	    select_variable_list=np.random.choice(variable_list, num_variable, replace=False)## sample variables
	    data_entry.append(",".join(select_variable_list))##save in variable name
	    ### list of variable name
	    var_name_list=[]
	    for variable in select_variable_list:
	        var_name=gen_var_name(subset_field,variable)
	        var_name_list.append(var_name)#saved in list
	    data_entry.append(",".join(var_name_list))
	    data_all.append(data_entry)
	data_gen_CSL=pd.DataFrame(data_all,columns=["dataset","var","var_name"])
	#data_gen_CSL
	for i in range(n_sample // 3):
	    data_gen_CSL["var_name"][i]=None
	for i in range(n_sample // 3, n_sample // 3 * 2):
	    data_gen_CSL["var_name"][i]=None
	    data_gen_CSL["var"][i]="all_variables"
	#print(data_gen_CSL.head())
	data_gen_CSL.to_csv(path + "CSL_30_v2.csv")

	## ATE
	data_all=[]
	for i in random_field_list:
	    data_entry=[]
	    field_name=field_list[i]
	    data_entry.append(field_name)##generate field_name
	    subset_field=df_variable_name[df_variable_name['field']==field_name]
	    variable_list=subset_field["variable"].unique()
	    select_variable=np.random.choice(variable_list, 2, replace=False)## sample two variable
	    treatment=select_variable[0]
	    data_entry.append(treatment)##generate treatment
	    subset_trt=subset_field[subset_field['variable']==treatment]
	    trt_name_list=subset_trt["name"].item().split(",")
	    trt_name=random.sample(trt_name_list,1)[0]
	    data_entry.append(trt_name)##treatment_name
	    #####outcome
	    outcome=select_variable[1]
	    data_entry.append(outcome)##generate outcome
	    subset_out=subset_field[subset_field['variable']==outcome]
	    out_name_list=subset_out["name"].item().split(",")
	    out_name=random.sample(out_name_list,1)[0]
	    data_entry.append(out_name)##outcomename

	    data_all.append(data_entry)
	data_gen=pd.DataFrame(data_all,columns=["dataset","treatment","treatment_name","outcome","outcome_name"])
	#print(data_gen)
	random_mask(data_gen, ["treatment_name"], 0.5)
	random_mask(data_gen, ["outcome_name"], 0.5)
	print(data_gen)
	data_gen.to_csv(path + "ATE_30_v2.csv")

	## HTE
	categorical_list=df_variable_name["variable"][df_variable_name["category"]=="categorical"].to_list()
	proportion_list=[round(i,2) for i in np.random.random_sample(size=n_sample*3)]
	class_list=np.random.choice([0,1,2], n_sample*3, replace=True)

	data_all=[]
	for i in random_field_list:
	    data_entry=[]
	    field_name=field_list[i]
	    data_entry.append(field_name)##generate field_name
	    subset_field=df_variable_name[df_variable_name['field']==field_name]
	    variable_list=subset_field["variable"].unique()
	    #####select variables
	    num_variable=np.random.choice(range(3,len(variable_list)+1),1, replace=False)## number of variable requested
	    select_variable_list=np.random.choice(variable_list, num_variable, replace=False)## sample variables
	    ####treatment
	    treatment=select_variable_list[0]
	    data_entry.append(treatment)##generate treatment
	    subset_trt=subset_field[subset_field['variable']==treatment]
	    trt_name_list=subset_trt["name"].item().split(",")
	    trt_name=random.sample(trt_name_list,1)[0]
	    data_entry.append(trt_name)##treatment_name
	    #####outcome
	    outcome=select_variable_list[1]
	    data_entry.append(outcome)##generate outcome
	    subset_out=subset_field[subset_field['variable']==outcome]
	    out_name_list=subset_out["name"].item().split(",")
	    out_name=random.sample(out_name_list,1)[0]
	    data_entry.append(out_name)##outcome name
	    ##### condition
	    #var_condition_list=[]
	    #var_name_condition_list=[]
	    #for var in select_variable_list[2:]:
	    var = select_variable_list[2]
	    var_name=gen_var_name(subset_field,var)#generate var_name
	    var_class=create_condition_2(var,var_name)#contion for each
	    #var_condition_list.append(var_condition)#saved in list
	    #var_name_condition_list.append(var_name_condition)
	    #data_entry.append(",".join(var_condition_list))##
	    #data_entry.append(",".join(var_name_condition_list))## condition for variable name
	    data_entry.append(var)##
	    data_entry.append(var_name)## condition for variable name
	    data_entry.append(var_class)## condition for variable name
	    ###
	    data_all.append(data_entry)
	data_gen_HTE=pd.DataFrame(data_all,columns=["dataset","treatment","treatment_name","outcome","outcome_name","condition","condition_name", "condition_value"])
	#data_gen_HTE
	random_mask(data_gen_HTE, ["treatment_name"], 0.5)
	random_mask(data_gen_HTE, ["outcome_name"], 0.5)
	#print(data_gen_HTE.head())
	data_gen_HTE.to_csv(path + "HTE_30_v2.csv")


	## MA
	data_all=[]
	for i in random_field_list:
	    data_entry=[]
	    field_name=field_list[i]
	    data_entry.append(field_name)##generate field_name
	    subset_field=df_variable_name[df_variable_name['field']==field_name]
	    variable_list=subset_field["variable"].unique()
	    #####select variables
	    num_variable=np.random.choice(range(3,len(variable_list)+1),1, replace=False)## number of variable requested
	    select_variable_list=np.random.choice(variable_list, num_variable, replace=False)## sample variables
	    ####treatment
	    treatment=select_variable_list[0]
	    data_entry.append(treatment)##generate treatment
	    subset_trt=subset_field[subset_field['variable']==treatment]
	    trt_name_list=subset_trt["name"].item().split(",")
	    trt_name=random.sample(trt_name_list,1)[0]
	    data_entry.append(trt_name)##treatment_name
	    #####outcome
	    outcome=select_variable_list[1]
	    data_entry.append(outcome)##generate outcome
	    subset_out=subset_field[subset_field['variable']==outcome]
	    out_name_list=subset_out["name"].item().split(",")
	    out_name=random.sample(out_name_list,1)[0]
	    data_entry.append(out_name)##outcome name
	    ##### mediator
	    #var_condition_list=[]
	    #var_name_condition_list=[]
	    #for var in select_variable_list[2:]:
	    #    var_name=gen_var_name(subset_field,var)#generate var_name
	    #    var_condition_list.append(var)#saved in list
	    #    var_name_condition_list.append(var_name)
	    #data_entry.append(",".join(var_condition_list))##
	    #data_entry.append(",".join(var_name_condition_list))## condition for variable name
	    var = select_variable_list[2]
	    var_name = gen_var_name(subset_field,var)
	    data_entry.append(var)##
	    data_entry.append(var_name)## condition for variable name
	    ###
	    data_all.append(data_entry)
	data_gen_mediator=pd.DataFrame(data_all,columns=["dataset","treatment","treatment_name","outcome","outcome_name","mediator","mediator_name"])
	data_gen_mediator
	random_mask(data_gen_mediator, ["treatment_name"], 0.5)
	random_mask(data_gen_mediator, ["outcome_name"], 0.5)
	random_mask(data_gen_mediator, ["mediator_name"], 0.5)
	data_gen_mediator.to_csv(path + "MA_30_v2.csv")

	## CPL same as HTE
	categorical_list=df_variable_name["variable"][df_variable_name["category"]=="categorical"].to_list()
	proportion_list=[round(i,2) for i in np.random.random_sample(size=n_sample*3)]
	class_list=np.random.choice([0,1,2], n_sample*3, replace=True)

	data_all=[]
	for i in random_field_list:
	    data_entry=[]
	    field_name=field_list[i]
	    data_entry.append(field_name)##generate field_name
	    subset_field=df_variable_name[df_variable_name['field']==field_name]
	    variable_list=subset_field["variable"].unique()
	    #####select variables
	    num_variable=np.random.choice(range(3,len(variable_list)+1),1, replace=False)## number of variable requested
	    select_variable_list=np.random.choice(variable_list, num_variable, replace=False)## sample variables
	    ####treatment
	    treatment=select_variable_list[0]
	    data_entry.append(treatment)##generate treatment
	    subset_trt=subset_field[subset_field['variable']==treatment]
	    trt_name_list=subset_trt["name"].item().split(",")
	    trt_name=random.sample(trt_name_list,1)[0]
	    data_entry.append(trt_name)##treatment_name
	    #####outcome
	    outcome=select_variable_list[1]
	    data_entry.append(outcome)##generate outcome
	    subset_out=subset_field[subset_field['variable']==outcome]
	    out_name_list=subset_out["name"].item().split(",")
	    out_name=random.sample(out_name_list,1)[0]
	    data_entry.append(out_name)##outcome name
	    ##### condition
	    #var_condition_list=[]
	    #var_name_condition_list=[]
	    #for var in select_variable_list[2:]:
	    var = select_variable_list[2]
	    var_name=gen_var_name(subset_field,var)#generate var_name
	    var_class=create_condition_2(var,var_name)#contion for each
	    #var_condition_list.append(var_condition)#saved in list
	    #var_name_condition_list.append(var_name_condition)
	    #data_entry.append(",".join(var_condition_list))##
	    #data_entry.append(",".join(var_name_condition_list))## condition for variable name
	    data_entry.append(var)##
	    data_entry.append(var_name)## condition for variable name
	    data_entry.append(var_class)## condition for variable name
	    ###
	    data_all.append(data_entry)
	data_gen_CPL=pd.DataFrame(data_all,columns=["dataset","treatment","treatment_name","outcome","outcome_name","condition","condition_name", "condition_value"])
	#data_gen_HTE
	random_mask(data_gen_CPL, ["treatment_name"], 0.5)
	random_mask(data_gen_CPL, ["outcome_name"], 0.5)
	#print(data_gen_HTE.head())
	data_gen_CPL.to_csv(path + "CPL_30_v2.csv")