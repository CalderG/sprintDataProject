import pandas as pd
import json
import os
# Establish list to put yearly dictionaries in and dataframe to put the list in?
sprintData = []
sprintDataFrame = pd.DataFrame()

# Establish results string for each year
yearResults = [f'{i}results.json' for i in range(1948, 2026)]
# Load in the given year's json file
for each_year in yearResults:
    with open(each_year, 'r') as f:
        data = json.load(f)
        # Extract the body/results from the dictionary
        yearlyDict = data['templates'][0]["divs"][0]["tables"][0]["body"]
        # At each year's dictionary, remove the 'classes', listIndex', and rowNum columns from the dictionaries
        for idx in range(len(yearlyDict)):
            yearlyDict[idx].pop("classes")
            yearlyDict[idx].pop("listIndex")
            try:
                yearlyDict[idx].pop("rowNum")
            except:
                print("No rowNum column present for that year") 
            yearlyDict[idx] = {key: [value] for key, value in yearlyDict[idx].items()}
            sprintData.append(pd.DataFrame.from_dict(yearlyDict[idx]))
        # Convert each entry into a row-like format for the dictionary, then turn it into a dataframe
        # Then append the dataframe to the list

# Add list of dataframes to empty dataframe
sprintDataFrame = pd.concat(sprintData)

# Export to CSV
sprintDataFrame.to_csv("sprintTestFrame.csv")








