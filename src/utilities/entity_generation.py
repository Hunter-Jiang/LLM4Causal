import numpy as np
import pandas as pd
import openai
import os
import pickle
from tqdm import tqdm
from io import StringIO

from utilities import openai_apis 

prompt_p1 = """
Instruction:
Your task is to list a field and then followed by a list of variable names related to that field.

The field can be a boarder area like economics, or a topic like air pollution.
Once the field is generated, please find some variable names that is measureable (either numerical or categorical) and related to the field.
Different variables should be saperated by comma.

Examples:
------------------------------------------------------
------------------------------------------------------
"""

prompt_p2 = """
------------------------------------------------------
------------------------------------------------------

Query:
Based on the instruction and examples, please generate 40 such field-variables for real-world data collection.
Please answer using markdown format and start with:
| field | variables |
"""

examples = [
    '• field: health \n• variables: blood pressure, weight, height, diet, drug use, alcohol use.',
    '• field: economic growth \n• variables: population, education, GDP, natural resources, health coverage.',
    '• field: air pollution \n• variables: PM25, gas consumption, agriculture, industry, mining.',
    '• field: wages \n• variables: salary, education, work experience, location, performance.',
   ]

def generate_entity_once():
    prompt = prompt_p1 + "\n\n".join(np.random.choice(examples, 4, replace = False)) + prompt_p2
    #print(prompt)
    curr_result = openai_apis.get_completion(prompt)
    d_read = pd.read_csv(
        StringIO(curr_result),
        sep='|',
        index_col=None
    ).iloc[1:,]
    d_read = d_read[d_read.columns[1:3]]
    d_read.columns = ["Input", "Output"]
    return d_read

prompt1_p1 = """
Instruction:
Your task is to provide possible variable names and ranges for programming for a given meaning.

Use commas to separate different name options, and classify the range into one of following: ["continuous", "categorical"].

Outputs should have two lines and one for each (Names and Range).

Examples:
------------------------------------------------------
------------------------------------------------------
"""

prompt1_p2 = """
------------------------------------------------------
------------------------------------------------------

Query:
Based on the instruction and examples, please generate 5 such names for the given input (meaning).

Meaning: 
"""

examples1 = [
    'Meaning: Solar energy capacity | Names: energy_capacity, capacity, solar_energy_capcity, capacity_solar \n Range: continuous',
    'Meaning: Internet of Things devices installed | Names: installed, device_installed, IoT_installed \n Range: continuous',
    'Meaning: Air quality index | Names: index, air_quality, quality_index, quality \n continuous',
    'Meaning: Gender | Names: gender, sex, gender_identity | categorical',
    'Input'
   ]

def generate_name_for_x(x):
    prompt = prompt1_p1 + "\n\n".join(examples1) + prompt1_p2 + x
    #print(prompt)
    curr_result = openai_apis.get_completion(prompt)
    try:
        tmp = curr_result.split("\n")
        name = tmp[0][6:].strip()
        rang = tmp[1][6:].strip()
        if rang not in ["continuous", "categorical"]:
            print("Warning: changing range from ", rang, "into continuous")
            rang = "continuous"
    except:
        name = ""
        rang = "err"
    return name, rang

if __name__ == '__main__':
    out = generate_name_for_x("color")
    print(out)