from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

from causaldm.learners.CEL.MA.ME_Single import ME_Single
from causaldm.learners.CPL13.disc import QLearning

import lingam
import time
import numpy as np 
import pandas as pd 

## CSL
def solve_CSL_function(X_causal, max_iter = 5000):
    model_lingam = lingam.ICALiNGAM(random_state = None, max_iter = max_iter)
    try:
        model_lingam.fit(X_causal)
        lingam_est = model_lingam.adjacency_matrix_.T
        lingam_est = np.asarray(lingam_est)
    except:
        print("LINGAM found no link")
        lingam_est = np.zeros((X_causal.shape[1], X_causal.shape[1]))
    return lingam_est

## ATE: DR estiamtes
def solve_ATE_function(df, A_col, Y_col):
    # soft_match
    A_col, Y_col = soft_match([A_col, Y_col], df.columns)
    # get X_cols
    X_cols = [x for x in df.columns if x not in [A_col, Y_col]]
    # propensity scores
    ps_model = LogisticRegression(
        solver='liblinear', 
        random_state=0, 
        fit_intercept = False
        ).fit(df[X_cols], df[A_col])
    ps = ps_model.predict_proba(df[X_cols])[:, 1]
    #print("ps coef", ps_model.intercept_, ps_model.coef_)
    # outcome imputation
    mu0 = LinearRegression().fit(
        df.query(f"{A_col}==0")[X_cols], 
        df.query(f"{A_col}==0")[Y_col]
        ).predict(df[X_cols])
    mu1 = LinearRegression().fit(
        df.query(f"{A_col}==1")[X_cols], 
        df.query(f"{A_col}==1")[Y_col]
        ).predict(df[X_cols])
    # get DR estimates
    return (
        np.mean(df[A_col]*(df[Y_col] - mu1)/ps + mu1) -
        np.mean((1-df[A_col])*(df[Y_col] - mu0)/(1-ps) + mu0)
    )

## HTE: S-learner
def solve_HTE_function(df, A_col, Y_col, condition):
    # soft_match
    ret = soft_match([A_col, Y_col, condition[0][0]], df.columns)
    A_col = ret[0]; Y_col = ret[1]; condition[0] = (ret[2], condition[0][1])
    # get X_cols
    XA_cols = [x for x in df.columns if x not in [Y_col]]
    X_col = [x for x in df.columns if x not in [A_col, Y_col]]
    # fit S-learner
    S_learner = LinearRegression()
    S_learner.fit(df[XA_cols], df[Y_col])
    # predict S-learner
    condition_0 = df[XA_cols].copy()
    condition_1 = df[XA_cols].copy()
    for df_pred in [condition_0, condition_1]:
        for col in X_col:
            df_pred[col] = 1
    condition_0[A_col] = 0
    condition_1[A_col] = 1
    # set the conditions
    for cond in condition:
        condition_0[cond[0]] = cond[1]
        condition_1[cond[0]] = cond[1]
    # get estiamtes
    HTE_S_learner = S_learner.predict(condition_1) - S_learner.predict(condition_0)
    #print("HTE estimates", np.mean(HTE_S_learner))
    return np.mean(HTE_S_learner)

## MA
def control_policy(state = None, dim_state=None, action=None, get_a = False):
    if get_a:
        action_value = np.array([0])
    else:
        state = np.copy(state).reshape(-1,dim_state)
        NT = state.shape[0]
        if action is None:
            action_value = np.array([0]*NT)
        else:
            action = np.copy(action).flatten()
            if len(action) == 1 and NT>1:
                action = action * np.ones(NT)
            action_value = 1-action
    return action_value

def target_policy(state, dim_state = 1, action=None):
    state = np.copy(state).reshape((-1, dim_state))
    NT = state.shape[0]
    pa = 1 * np.ones(NT)
    if action is None:
        if NT == 1:
            pa = pa[0]
            prob_arr = np.array([1-pa, pa])
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            raise ValueError('No random for matrix input')
    else:
        action = np.copy(action).flatten()
        action_value = pa * action + (1-pa) * (1-action)
    return action_value

def solve_MA_function(data, treatment, response, mediator):
    # soft_match
    treatment, response, mediator = soft_match([treatment, response, mediator], data.columns)
    # get vectors
    states = data[[x for x in data.columns if x not in [treatment, response, mediator]]]
    states.columns = ["states"]
    action = data[[treatment]]
    mediator = data[[mediator]]
    reward = data[[response]]

    # estimate MA
    df = {'state':states,'action':action,'mediator':mediator,'reward':reward}
    problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)}
    Robust_est = ME_Single(df, r_model = 'OLS',
                         problearner_parameters = problearner_parameters,
                         truncate = 50,
                         target_policy=target_policy, control_policy = control_policy,
                         dim_state = 1, dim_mediator = 1,
                         MCMC = 50,
                         nature_decomp = True,
                         seed = 10,
                         method = 'Robust')

    Robust_est.estimate_DE_ME()
    #print(Robust_est.est_DE, Robust_est.est_ME, Robust_est.est_TE)
    return Robust_est.est_DE, Robust_est.est_ME, Robust_est.est_TE


## CPL
def solve_CPL_function(data, treatment, response, control):
    # soft_match
    ret = soft_match([treatment, response, control[0]], data.columns)
    treatment = ret[0]; response = ret[1]; control = (ret[2], control[1])
    #print(treatment, response, control)
    R = data[response]
    S = data[[x for x in data.columns if x.split("-")[0] in [control[0]]]]
    A = data[[x for x in data.columns if x.split("-")[0] in [treatment]]] 
    print(A.columns)
    
    # set up model
    S.columns = ["".join(x.split("-")) for x in S.columns]
    A.columns = ["".join(x.split("-")) for x in A.columns]
    #print(R, S, A)
    
    # initialize the learner
    QLearn = QLearning.QLearning()
    # specify the model you would like to use
    # If want to include all the variable in S and A with no specific model structure, then use "Y~."
    # Otherwise, specify the model structure by hand
    # Note: if the action space is not binary, use C(A) in the model instead of A
    model_info = [
        {
            "model": "Y ~ C(" + A.columns[0] + ") * (" + S.columns[0] + ")",
             'action_space':{A.columns[0]:[0,1]}
        },
        {
            "model": "Y ~ C(" + A.columns[1] + ") * (" + S.columns[1] + ")",
            'action_space':{A.columns[1]:[0,1]}
        }]
    # train the policy
    QLearn.train(S, A, R, model_info, T=2)

    # return optimal A
    newX = pd.DataFrame(columns = list(S.mean().index))
    newX.loc[0] = S.mean()
    newX.at[0, control[0] + "0"] = float(control[1])
    return QLearn.recommend_action(S, newX=newX).to_list()

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

## Resolve soft matche columns
def soft_match(input_cols, data_cols):
    data_cols = [x.split("-")[0] for x in data_cols]
    input_modified = []
    for column in input_cols:
        if column in data_cols:
            input_modified.append(column)
        else:
            flag = True
            if (column[-1] == "s") and (column[:-1] in data_cols):
                input_modified.append(column[:-1])
                continue
            if (column[-2:] == "es") and (column[:-2] in data_cols):
                input_modified.append(column[:-2])
                continue
            for column_match in data_cols:
                if column_match.find(column) > -1:
                    flag = False
                    input_modified.append(column_match)
                    break
            if flag:
                input_modified.append(column)
    return input_modified


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
    if json_in.find("</s>") > -1:
        json_in = json_in.split("</s>")[0]
    json_eval = eval(json_in)
    data = pd.read_csv(saved_dir + json_eval["dataset"][0])
    data = data[[x for x in data.columns if x.find("Unnamed") == -1]]
    if json_eval["causal_problem"] == ['CSL', None]:
        ret = solve_CSL_function(data)
    elif json_eval["causal_problem"] == ['CEL', "ATE"]:
        ret = solve_ATE_function(data, json_eval["treatment"][0], json_eval["response"][0])
    elif json_eval["causal_problem"] == ['CEL', "HTE"]:
        ret = solve_HTE_function(data, json_eval["treatment"][0], json_eval["response"][0], json_eval["condition"])
    elif json_eval["causal_problem"] == ['CEL', "MA"]:
        ret = solve_MA_function(data, json_eval["treatment"][0], json_eval["response"][0], json_eval["mediator"][0])
    elif json_eval["causal_problem"] == ['CPL', None]:
        ret = solve_CPL_function(
            data, 
            json_eval["treatment"][0], 
            json_eval["response"][0], 
            json_eval["condition"][0]
            )
    return ret, data

action_classes = ['0', '1', '2', 'A', 'B', 'C', 'I', 'II', 'III']
def generate_function_outcome(JSON_output, function_output, data):
    if JSON_output.find("</s>") > -1:
        JSON_output = JSON_output.split("</s>")[0]
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