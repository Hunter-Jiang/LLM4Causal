### difference: exclude outputs having two variable names
import pandas as pd
import numpy as np
import re

def split_par(str_in):
  if str_in.find("(") == -1:
    ret = (str_in.strip(), "")
  else:
    re_res = re.split(r'\(|\)', str_in)
    ret = (re_res[0].strip(), re_res[1].strip())
  return ret

def split_check_objects(str_in):
  spl = str_in.split(":")
  ret = []
  match spl[0]:
    case "- Interested variable":
      spl_1 = spl[1].split(",")
      for var in spl_1:
        ret_1 = split_par(var)
        ret.append(ret_1)
    case "- Treatment variable":
      ret_1 = split_par(spl[1])
      ret.append(ret_1)
    case "- Outcome variable":
      ret_1 = split_par(spl[1])
      ret.append(ret_1)
    case "- Mediator variable":
      ret_1 = split_par(spl[1])
      ret.append(ret_1)
    case "- Dataset name":
      ret.append((spl[1].strip(), ""))
    case "- Group Variable":
      ret.append((spl[1].strip(), ""))
    case "- Group Condition":
      re_res = re.split(r'\(|\)', spl[1])
      ret.append((re_res[0].strip(), ""))
      ret.append((re_res[1].strip().split("=")[0], ""))
    case "- Condition variable":
      ret.append((spl[1].strip(), ""))
    case "- Condition value":
      re_res = re.split(r'\(|\)', spl[1])
      ret.append((re_res[0].strip(), ""))
      ret.append((re_res[1].strip().split("=")[0], ""))
    case _: #
      pass

  return ret

def check_input(input_raw, check_list, idx):
  for check in check_list:
    if check[0] == "all variables":
      continue
    pos = input_raw.lower().find(check[0].lower())
    try:
      assert pos > -1
    except:
      print(idx, check, input_raw)
      assert pos > -1
    pos1 = input_raw.lower().find(check[1].lower())
    if pos1 == -1:
      pos += len(check[0])
      input_raw = input_raw[:pos] + " (" + check[1] + ")" + input_raw[pos:]
      #print(input_raw)

    pos = input_raw.lower().find(check[0].lower())
    try:
      assert pos > -1
    except:
      print(idx, check, input_raw)
      assert pos > -1
    pos = input_raw.lower().find(check[1].lower())
    try:
      assert pos > -1
    except:
      print(idx, check, input_raw)
      assert pos > -1
  return input_raw

def change_csl(df_in):
	df_in.at[0, "input"] = "Are there any discernible effects on median home price and vacancy rates within the housing_market.csv dataset?"
	df_in.at[1, "input"] = "In the housing_market.csv dataset, what causal connections exist between the median home price and vacancy rates?"
	df_in.at[3, "input"] = "Can the housing_market.csv dataset provide evidence of any causal relationships between median home price and vacancy rates?"
	df_in.at[4, "input"] = "Within the housing_market.csv dataset, what causal links can be determined between median home price and vacancy rates?"
	df_in.at[5, "input"] = "Is there evidence in the political_engagement.csv dataset indicating a significant impact of legislation passed, voter turnout, and political rallies attendance on one another?"
	df_in.at[6, "input"] = "Within the political_engagement.csv dataset, can you outline the causal connections among legislation passed, voter turnout, and political rallies attendance"
	df_in.at[7, "input"] = "Could you identify any direct relationships within the political_engagement.csv dataset where one variable influences another among legislation passed, voter turnout, and political rallies attendance?"
	df_in.at[8, "input"] = "Can we determine the presence of causal effects among legislation passed, voter turnout, and political rallies attendance using the political_engagement.csv dataset?"
	df_in.at[9, "input"] = "Can the political_engagement.csv dataset reveal any direct relationships between legislation passed, voter turnout, and political rallies attendance?"
	df_in.at[16, "input"] = "Can we determine the existence of causal effects between the variables in the retail_sales.csv dataset?"
	df_in.at[20, "input"] = "In the consumer_electronics.csv dataset, what are the causal pathways among user satisfaction (user_satisfaction), the monthly sales volume (monthly_sales), and market share (sector_market_share)?"
	df_in.at[21, "input"] = "Can we identify any causal connections among user satisfaction (user_satisfaction), the monthly sales volume (monthly_sales), and market share (sector_market_share) from the consumer_electronics.csv dataset?"
	df_in.at[22, "input"] = "Are there any causal effects documented in the consumer_electronics.csv dataset among user satisfaction (user_satisfaction), the monthly sales volume (monthly_sales), and market share (sector_market_share)?"
	df_in.at[23, "input"] = "Looking at the data from consumer_electronics.csv, is there a way to establish causal effects among user satisfaction (user_satisfaction), the monthly sales volume (monthly_sales), and market share (sector_market_share)?"
	df_in.at[24, "input"] = "Within the consumer_electronics.csv dataset, what is the nature of the causal connections among sales volume (monthly_sales), user satisfaction (user_satisfaction), and market share (sector_market_share)?"
	df_in.at[25, "input"] = "Are there any causal effects between campaign donations (donation_amount) and legislation passed (bills_passed) in the political_engagement.csv dataset?"
	df_in.at[26, "input"] = "Does the political_engagement.csv dataset provide evidence of a direct link between building campaign donations (donation_amount) and legislation passed (bills_passed)?"
	df_in.at[27, "input"] = "Can you identify the presence of causal links between campaign donations (donation_amount) and legislation passed (bills_passed) in the political_engagement.csv dataset?"
	df_in.at[29, "input"] = "Are there any evident causal effects between campaign donations (donation_amount) and legislation passed (bills_passed) in the political_engagement.csv dataset?"

	return df_in

def change_ate(df_in):
	df_in.at[5, "input"] = "In the political_engagement.csv dataset, how significant is the role of having a party membership (member_party) in the number of legislation passed or bills_passed?"
	df_in.at[6, "input"] = "Can we see a noticeable difference in the legislation passed, as measured by bills_passed, linked to the presence of party membership noted as member_party in the political_engagement.csv?"
	df_in.at[7, "input"] = "How does holding a party membership (member_party) status within the political_engagement.csv dataset translate to the actual enactment of legislation passed (bills_passed)?"
	df_in.at[8, "input"] = "In the context of the political_engagement.csv, what is the magnitude of influence that party membership (member_party) status exerts on the count of legislation passed (bills_passed)?"
	df_in.at[9, "input"] = "With the political_engagement.csv data, what degree of change in the quantity of legislation passed (bills_passed) can we observe when comparing individuals with and without a party membership, or member_party, designation?"
	df_in.at[15, "input"] = "Can an increase in retail revenue lead to a significant change in the store count, as shown in the retail_sales.csv data?"
	df_in.at[20, "input"] = "How significantly does user satisfaction (satisfaction_score) influence the sales volume (volume) according to the consumer_electronics.csv data?"
	df_in.at[25, "input"] = "Can we see a significant change in the number of legislation passed due to variations in voter turnout as evidenced by the political_engagement.csv dataset?"
	df_in.at[26, "input"] = "In the data from political_engagement.csv, how much does voter turnout contribute to the enactment of legislation passed?"
	df_in.at[28, "input"] = "To what extent does voter turnout drive the legislation passed getting passed as shown in the political_engagement.csv?"
	return df_in

def change_hte(df_in):
	df_in.at[0, "input"] = "What is the influence of varying mortgage rates (interest_rate) on vacancy rates in the housing_market.csv dataset, specifically when the median home price (median_home_price) is set at 0.93?"
	df_in.at[1, "input"] = "In cases where the median home price (median_home_price) stands at 0.93, can we observe a significant change in vacancy rates due to shifts in mortgage rates (interest_rate) according to the data in housing_market.csv?"
	df_in.at[2, "input"] = "How would an adjustment to mortgage rates (interest_rate) translate to changes in vacancy rates within housing_market.csv when we specifically look at homes with a median home price (median_home_price) equal to 0.93?"
	df_in.at[3, "input"] = "Concerning the data in housing_market.csv, what magnitude of impact do changes in mortgage rates (interest_rate) have on vacancy rates when analyzing houses grouped by a median home price (median_home_price) of 0.93?"
	df_in.at[4, "input"] = "Focusing on the subgroup of properties with a median home price (median_home_price) of 0.93 in housing_market.csv, how does the modification of mortgage rates (interest_rate) reflect on the subsequent vacancy rates?"
	df_in.at[5, "input"] = "How does having a party membership (affiliation_party) influence the number of legislation passed according to the political_engagement.csv dataset, specifically when political rallies attendance is set at 0.26 (political_attendance=0.26)?"
	df_in.at[6, "input"] = "In the context of the political_engagement.csv, can we see a notable impact on the legislation passed attributable to party membership (affiliation_party) among those with a political rallies attendance score of 0.26 (political_attendance=0.26)?"
	df_in.at[7, "input"] = "What effect does party membership (affiliation_party) have on the effectiveness of the legislation passed in situations where political rallies attendance is consistently at 0.26 (political_attendance=0.26), as shown in the political_engagement.csv dataset?"
	df_in.at[8, "input"] = "Does party membership (affiliation_party) significantly alter the legislation passed within the political_engagement.csv when assessing groups with a political rallies attendance of 0.26 (political_attendance=0.26)?"
	df_in.at[9, "input"] = "Is there a discernible difference in the number of legislation passed due to party membership (affiliation_party) in the political_engagement.csv, under the condition where individuals attend political rallies attendance at a rate of 0.26 (political_attendance=0.26)?"
	df_in.at[11, "input"] = "In the context of the public_health.csv dataset, how does the introduction of disease incidence influence vaccination rates (vaccination_rate) when we focus on a subset with an life expectancy, or average_lifespan, of 0.77?"
	df_in.at[12, "input"] = "Can we see noticeable shifts in vaccination rates (vaccination_rate) as a result of varying disease incidence, when examining the data for a group with an life expectancy (average_lifespan) of 0.77, according to the public_health.csv?"
	df_in.at[13, "input"] = "When analyzing the public_health.csv data, what kind of difference does the disease incidence make on vaccination rates (vaccination_rate) among populations characterized by an life expectancy (average_lifespan) of 0.77?"
	df_in.at[15, "input"] = "How does consumer spending (customer_expenditure) respond to shifts in retail revenue in the retail_sales.csv data when product demand (demand_level) is held steady at a level of 0.85?"
	df_in.at[16, "input"] = "Can we observe significant changes in consumer spending (customer_expenditure) as a result of varying retail revenue if we look at cases in the retail_sales.csv where product demand (demand_level) remains fixed at 0.85?"
	df_in.at[17, "input"] = "When examining the retail_sales.csv dataset, what degree of change is noticed in consumer spending (customer_expenditure) when retail revenue is adjusted, given that the product demand (demand_level) is set at 0.85?"
	df_in.at[18, "input"] = "In situations where the retail_sales.csv dataset shows a product demand level (demand_level) of 0.85, what can we infer about the influence of retail revenue on consumer spending (customer_expenditure)?"
	df_in.at[19, "input"] = "Across the retail_sales.csv dataset, with product demand constrained to a demand level (demand_level) of 0.85, what impacts on consumer spending (customer_expenditure) are detected when retail revenue experiences changes?"
	df_in.at[20, "input"] = "How does the innovation rate influence the total number of product releases (total_releases) in the consumer_electronics.csv dataset when user satisfaction (customer_satisfaction) is held at a level of 0.46?"
	df_in.at[21, "input"] = "In light of the consumer_electronics.csv data, can we see a noticeable impact on product releases (total_releases) when the innovation rate is adjusted, given that user satisfaction (customer_satisfaction) remains constant at 0.46?"
	df_in.at[22, "input"] = "What effect does varying the innovation rate have on the volume of product releases (total_releases) recorded in the consumer_electronics.csv, under the specific condition where user satisfaction (customer_satisfaction) is at 0.46?"
	df_in.at[23, "input"] = "When examining the consumer_electronics.csv, what can be inferred about the change in product releases (total_releases) as a result of shifts in the innovation rate, while maintaining user satisfaction (customer_satisfaction) at the 0.46 benchmark?"
	df_in.at[24, "input"] = "Considering the consumer_electronics.csv, how might the innovation rate drive changes in product releases (total_releases), especially when taking into account a user satisfaction (customer_satisfaction) score of 0.46?"
	df_in.at[25, "input"] = "How does the voter turnout influence campaign donations when the legislation passed (laws_enacted) measure is at 0.85, as shown in the political_engagement.csv dataset?"
	df_in.at[26, "input"] = "Can we observe a noticeable change in campaign donations due to varying levels of voter turnout within the political_engagement.csv file, specifically when legislation passed (laws_enacted) is fixed at 0.85?"
	df_in.at[27, "input"] = "Looking at the data in political_engagement.csv, what can we say about the impact of voter turnout on campaign donations when the group variable of legislation passed (laws_enacted) remains at the level of 0.85?"
	df_in.at[28, "input"] = "From the information in political_engagement.csv, what effect does voter turnout have on the amount of campaign donations received, given that legislation passed (laws_enacted) is at a steady rate of 0.85?"
	df_in.at[29, "input"] = "According to the political_engagement.csv, how are campaign donations affected by changes in voter turnout, while keeping the condition of legislation passed (laws_enacted) at a constant value of 0.85?"
	return df_in

def change_ma(df_in):
	df_in.at[2, "input"] = "How can the housing_market.csv dataset help us estimate the magnitude of the impact of mortgage rates (interest_rate) on the median home price, while considering the role of the vacancy rates as a mediator?"
	df_in.at[3, "input"] = "Based on the housing_market.csv dataset, can we estimate the extent to which vacancy rates mediates the relationship between the treatment variable mortgage rates (interest_rate) and the outcome variable median home price?"
	df_in.at[5, "input"] = "Within the political_engagement.csv dataset, to what extent does the voter turnout (participation_rate) mediates the effect of political rallies attendance on the campaign donations?"
	df_in.at[6, "input"] = "How significantly does the voter turnout (participation_rate) function as a go-between for political rallies attendance and its influence on campaign donations, as seen in the political_engagement.csv information?"
	df_in.at[7, "input"] = "Can we determine the direct causal impact of political rallies attendance on the campaign donations in the political_engagement.csv dataset, and how significant is the role of voter turnout (participation_rate) as a mediator in this relationship?"
	df_in.at[10, "input"] = "How much of the effect of the treatment variable, life expectancy, on the outcome variable, hospital beds per capita, is mediated by the mediator variable, obesity rates, within the public_health.csv dataset?"
	df_in.at[11, "input"] = "Within the context of the public_health.csv data, what is the size of the mediator effect of the obesity rates in mediating the relationship between life expectancy and the outcome of the hospital beds per capita?"
	df_in.at[12, "input"] = "What is the extent to which obesity rates mediates the relationship between the treatment variable, life expectancy, and the outcome viable, the hospital beds per capita, in the public_health.csv dataset?"
	df_in.at[15, "input"] = "How much of the effect of product demand on retail revenue can be attributed to the mediating factor of consumer spending (expenditure), as revealed by the retail_sales.csv dataset?"
	df_in.at[16, "input"] = "In the retail_sales.csv dataset, what is the proportion of the impact of product demand on retail revenue that is explained by the mediator variable of consumer spending (expenditure)?"
	df_in.at[19, "input"] = "How influential is the role of the consumer spending (expenditure) as a mediator in linking the product demand to the retail revenue, according to the retail_sales.csv dataset?"
	df_in.at[21, "input"] = "Within the consumer_electronics.csv dataset, how significant is the mediated effect of product releases (releases) in explaining the relationship between the treatment variable innovation rate (innovation_index) and the outcome variable market share (sector_market_share)?"
	df_in.at[25, "input"] = "By analyzing the political_engagement.csv data, how can we quantify the specific contribution of the mediator variable, voter turnout, in explaining the effect of campaign donations (campaign_donations) on the political rallies attendance (attendance_numbers)?"
	df_in.at[26, "input"] = "In the analysis of political_engagement.csv, can we determine the extent to which voter turnout serves as a bridge between campaign donations (campaign_donations) and the number of political rallies attendance (attendance_numbers)?"
	df_in.at[27, "input"] = "How significant is the role of voter turnout as a mediator in the causal pathway from campaign donations (campaign_donations) to the political rallies attendance (attendance_numbers) in the political_engagement.csv dataset?"
	df_in.at[28, "input"] = "To what extent does the voter turnout mediate the relationship between campaign donations (campaign_donations) and the political rallies attendance (attendance_numbers), as indicated by the political_engagement.csv dataset?"
	df_in.at[29, "input"] = "In the political_engagement.csv dataset, what is the extent of the mediator effect of voter turnout on the relationship between the campaign donations (campaign_donations) and political rallies attendance (attendance_numbers)?"
	return df_in

def change_cpl(df_in):
	df_in.at[0, "input"] = "Looking at the housing_market.csv data, when the vacancy rates is at 0.07 (empty_units_rate=0.07), what should be the ideal median home price (home_price) to aim for that could influence the mortgage rates (home_loan_rate) favorably?"
	df_in.at[2, "input"] = "In light of the housing_market.csv, what median home price (home_price) adjustment would be most prudent to potentially counteract the effects on mortgage rates (home_loan_rate) when dealing with a vacancy rates of 0.07 (empty_units_rate=0.07)?"
	df_in.at[4, "input"] = "Using the housing_market.csv as a reference, what guidance is there on setting an appropriate median home price (home_price) to potentially respond to mortgage rates (home_loan_rate) when faced with a vacancy rates challenge of 0.07 (empty_units_rate=0.07)?"
	df_in.at[5, "input"] = "Could you guide me on how to adjust the number of political rallies attendance (rallies_attended) as suggested by the political_engagement.csv, particularly when considering that the rate of legislation passed (laws_enacted) is at 0.57, to maximize the amount of campaign donations (funds_received)?"
	df_in.at[7, "input"] = "With the information provided by political_engagement.csv and the condition that legislation passed (laws_enacted) is 0.57, what's the best course of action to take regarding the frequency of political rallies attendance (rallies_attended) to potentially influence campaign donations (funds_received)?"
	df_in.at[8, "input"] = "What's the advisable approach on political rallies attendance (rallies_attended), according to the insights from political_engagement.csv, to possibly affect the campaign donations (funds_received) when the benchmark for legislation passed (laws_enacted) is at 0.57?"
	df_in.at[9, "input"] = "Given that the legislation passed (laws_enacted) is currently at a value of 0.57, as per the political_engagement.csv dataset, what steps should be considered when planning political rallies attendance (rallies_attended) to possibly have a positive impact on the subsequent campaign donations (funds_received)?"
	df_in.at[10, "input"] = "If someone has an obesity rates of 0.73 (rate_of_obesity=0.73) as noted in our public_health.csv dataset, what course of action would you suggest to improve their life expectancy (lifespan) with respect to the observed vaccination rates?"
	df_in.at[11, "input"] = "How should we adjust our approach to enhancing life expectancy (lifespan) in the context of the public_health.csv data, particularly for individuals with an obesity rates at 0.73 (rate_of_obesity=0.73), in order to effectively influence vaccination rates?"
	df_in.at[12, "input"] = "Can the public_health.csv file provide guidance on the best steps to take in increasing life expectancy (lifespan), given that the obesity rates are currently standing at 0.73 (rate_of_obesity=0.73), with an aim to positively impact vaccination rates?"
	df_in.at[13, "input"] = "Considering the public_health.csv findings, what life expectancy (lifespan) improvements should be prioritized for individuals with an obesity rates of 0.73 (rate_of_obesity=0.73) to potentially affect their vaccination rates?"
	df_in.at[14, "input"] = "With obesity rates at 0.73 (rate_of_obesity=0.73) according to the public_health.csv, what recommended actions could be taken to lengthen life expectancy (lifespan) that might, in turn, have a beneficial effect on vaccination rates?"
	df_in.at[20, "input"] = "In light of an innovation rate standing at 0.69 (innovation_level=0.69) within the consumer_electronics.csv dataset, what guidance can be offered regarding the timing or intensity of product releases to potentially enhance sales volume (units_sold)?"
	df_in.at[21, "input"] = "Can you advise on how to modify our approach to product releases to improve sales volume (units_sold), based on the data from consumer_electronics.csv, especially considering our current innovation rate is at 0.69 (innovation_level=0.69)?"
	df_in.at[22, "input"] = "With the consumer_electronics.csv data at hand, and acknowledging an innovation rate of 0.69 (innovation_level=0.69), what product releases strategy might be optimal for maximizing sales volume?"
	df_in.at[23, "input"] = "When considering the consumer_electronics.csv dataset, how would you suggest we adjust the frequency or nature of our product releases in response to an innovation rate measured at 0.69 (innovation_level=0.69), with the aim of boosting sales volume (units_sold)?"
	df_in.at[24, "input"] = "Looking at the evidence from consumer_electronics.csv, what actions related to product releases would you recommend for achieving the best sales volume (units_sold), given that our innovation rate hovers around 0.69 (innovation_level=0.69)?"
	df_in.at[25, "input"] = "What guidance can you offer, after looking into the political_engagement.csv dataset, on how an individual or group should approach political rallies attendance to influence the number of legislation passed, particularly when their campaign donations are at 0.49 (campaign_donations=0.49)?"
	df_in.at[26, "input"] = "Considering the political_engagement.csv data, what action should one take regarding their political rallies attendance to potentially affect the legislation passed when accounting for a campaign donations level of 0.49 (campaign_donations=0.49)?"
	df_in.at[27, "input"] = "What steps should an organization consider with respect to political rallies attendance to maximize the impact on legislation passed, as per insights from the political_engagement.csv, given that their campaign donations are set at the value of 0.49 (campaign_donations=0.49)?"
	df_in.at[28, "input"] = "In light of the information from the political_engagement.csv, what would be the most effective strategy for someone to undertake in terms of political rallies attendance to contribute towards the legislation passed, bearing in mind that their campaign donations are currently 0.49 (campaign_donations=0.49)?"
	df_in.at[29, "input"] = "When analyzing the political_engagement.csv dataset, can you advise on the level of activity in political rallies attendance one should strive for to have a positive effect on legislation passed under the circumstance where their campaign donations stand at 0.49 (campaign_donations=0.49)?"
	return df_in

if __name__ == '__main__':
	files = ["CSL", "ATE", "HTE", "MA", "CPL"]
	path = "../data/run_files/"

	for file in files:
	  # read
	  fn = path + "p30_" + file + "_v2_check.csv"
	  df = pd.read_csv(fn)
	  # change
	  match file:
	    case "CSL":
	      df = change_csl(df)# split
	      df_use = df[["prompt", "input", "output", "var"]]
	    case "ATE":
	      df = change_ate(df)
	      df_use = df[["prompt", "input", "output"]]
	    case "HTE":
	      df = change_hte(df)
	      df_use = df[["prompt", "input", "output"]]
	    case "MA": 
	      df = change_ma(df)
	      df_use = df[["prompt", "input", "output"]]
	    case "CPL":
	      df = change_cpl(df)
	      df_use = df[["prompt", "input", "output"]]
	    case _:
	      raise NotImplementedError

	  for idx, row in df_use.iterrows():
	    check = []
	    for line in row["prompt"].split("\n"):
	      if line[:2] == "--":
	        break
	      else:
	        check += split_check_objects(line)
	    # get interested parts
	    df_use.at[idx, "checked_input"] = check_input(row["input"], check, idx)
	    if df_use.at[idx, "checked_input"].find(") (") > -1:
	      print("-"*100)
	      print(idx)
	      print(df_use.at[idx, "output"])
	      print(df_use.at[idx, "checked_input"])

	  print("Output " + "p30_sanity_" + file + ".csv")
	  df_use.to_csv(path + "p30_" + file + "_V2_sanity_.csv")
	  print("-" * 100)

