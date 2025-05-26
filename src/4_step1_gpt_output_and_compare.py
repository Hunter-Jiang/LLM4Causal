import json
import os
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

import pandas as pd 
import numpy as np 

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(messages, tools=None, tool_choice=None, model="gpt-4-1106-preview", temp = 0):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages, "temperature": temp}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

def get_tools():
    return [
    {
        "type": "function",
        "function": {
            "name": "causal_graph_learning",
            "description": "Return the causal structure from a dataset with variables of interest",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "The name of the input dataset",
                    },
                    "nodes": {
                        "type": "string",
                        "description": "name of the interested variable saperated by commas, if no variable name is specified then put all_variables as the placeholder",
                    },
                },
                "required": ["dataset", "nodes"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "average_treatment_effect",
            "description": "Return the average treatment effect from causal effect learning, given a dataset and treatment / outcome variable name",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "The name of the input dataset",
                    },
                    "treatment": {
                        "type": "string",
                        "description": "a variable name of treatment in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "a variable name of outcome in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                },
                "required": ["dataset", "treatment", "outcome"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "heterogeneous_treatment_effect",
            "description": "Return the heterogeneous treatment effect from causal effect learning, given a dataset and treatment / outcome / condition variable name and a condition value",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "The name of the input dataset",
                    },
                    "treatment": {
                        "type": "string",
                        "description": "a variable name of treatment in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "a variable name of outcome in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "condition_variable": {
                        "type": "string",
                        "description": "a variable name of condition in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "condition_value": {
                        "type": "number",
                        "description": "the condition value of the condition_variable name",
                    },
                },
                "required": ["dataset", "treatment", "outcome", "condition_variable", "condition_value"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mediation_analysis",
            "description": "Return the mediation analysis from causal effect learning, given a dataset and treatment / outcome / mediater variable names",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "The name of the input dataset",
                    },
                    "treatment": {
                        "type": "string",
                        "description": "a variable name of treatment in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "a variable name of outcome in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "mediator": {
                        "type": "string",
                        "description": "a variable name of mediator in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                },
                "required": ["dataset", "treatment", "outcome", "mediator"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "causal_policy_learning",
            "description": "Return the best option of the treatment w.r.t a condition to get the best response suggested by a dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "The name of the input dataset",
                    },
                    "treatment": {
                        "type": "string",
                        "description": "a variable name of treatment in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "a variable name of outcome in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "condition_variable": {
                        "type": "string",
                        "description": "a variable name of condition in the study extracted from the input, denoting a possible column name from the dataset and connected by underscores",
                    },
                    "condition_value": {
                        "type": "number",
                        "description": "the condition value of the condition_variable name",
                    },
                },
                "required": ["dataset", "treatment", "outcome", "condition_variable", "condition_value"],
            },
        }
    },
]

def get_gpt_response(input_file, output_file, model, temp):
    data = pd.read_csv(input_file)
    data["name"] = None
    print(data.head())

    for idx, row in data.iterrows():
      if type(row["name"]) == str:
        print("skip row", idx)
        continue

      input_query = row["input"]
      messages = []
      messages.append({"role": "user",
                       "content": input_query})
      chat_response = chat_completion_request(
          messages, tools=tools, model = model, temp = temp
      )
      try:
        assistant_message = chat_response.json()["choices"][0]["message"]
        data.at[idx, "name"] = assistant_message["tool_calls"][0]["function"]["name"]
        data.at[idx, "arguments"] = assistant_message["tool_calls"][0]["function"]["arguments"]
      except:
        data.at[idx, "name"] = ""
        data.at[idx, "arguments"] = dict()
      print("-" * 50)
      print(row["output"])
      print(data.at[idx, "name"])
    print(data.head())

    data.to_csv(output_file)

def get_gpt_guided_response(input_file, output_file, model, temp):
    #assistant_message["tool_calls"][0]["function"]
    data = pd.read_csv(input_file)
    data["name"] = None
    print(data.head())
    for idx, row in data.iterrows():
      if type(row["name"]) == str:
        print("skip row", idx)
        continue

      gl = eval(row["output"])["causal_problem"]
      match gl:
        case ["CSL", None]:
          tool1 = tools[0]
        case ["CEL", "ATE"]:
          tool1 = tools[1]
        case ["CEL", "HTE"]:
          tool1 = tools[2]
        case ["CEL", "MA"]:
          tool1 = tools[3]
        case ["CPL", None]:
          tool1 = tools[4]

      input_query = row["input"]
      messages = []
      messages.append({"role": "user",
                       "content": input_query})
      chat_response = chat_completion_request(
          messages, 
          tools=[tool1], 
          tool_choice={"type": "function", "function": {"name": tool1["function"]["name"]}},
          model = model,
          temp = temp
      )
      try:
        assistant_message = chat_response.json()["choices"][0]["message"]
        data.at[idx, "name"] = assistant_message["tool_calls"][0]["function"]["name"]
        data.at[idx, "arguments"] = assistant_message["tool_calls"][0]["function"]["arguments"]
      except:
        data.at[idx, "name"] = ""
        data.at[idx, "arguments"] = dict()
      print("-" * 50)
      print(row["output"])
      print(data.at[idx, "name"])
    data.to_csv(output_file)

if __name__ in "__main__":
    with open("openai_key.txt", "rb") as handle:
        str_in = handle.readline()
    openai.api_key = str_in.decode('UTF-8')

    in_file = "../data/replicates/evaluate_p30.csv"
    models = ["gpt-4o", "gpt-4o-mini"]#"gpt-3.5-turbo"]#, "gpt-4-1106-preview"]
    temps = [0, 0.3]
    attempts = 3

    for GPT_MODEL in models:
        for temperature in temps:
            for att in range(attempts):
                tools = get_tools()

                TEMP = str(temperature).replace(".", "d")
                out_file = "../data/run_files/{}_run_{}_{}_p30.csv".format(
                        GPT_MODEL[:10], str(att), TEMP)
                out_file1 = "../data/run_files/{}_run_{}_{}_p30_guided.csv".format(
                        GPT_MODEL[:5], str(att), TEMP)
                if os.path.isfile(out_file):
                    print("skip", out_file)
                else:
                    print("working on", out_file)
                    get_gpt_response(
                        input_file = in_file,
                        output_file = out_file,
                        model = GPT_MODEL,
                        temp = temperature)
                #if os.path.isfile(out_file1):
                #    print("skip", out_file1)
                #else:
                #    print("working on", out_file1)
                #    get_gpt_guided_response(
                #        input_file = in_file,
                #        output_file = out_file1,
                #        model = GPT_MODEL,
                #        temp = temperature)
    



    