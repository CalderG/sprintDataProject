import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

editingSprintFrame = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/updated_clean_windaided.csv", low_memory = False)

altitudes2Add = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/additionalVenues.csv")


# Time Correction Function - credited from Jonas Mureika's paper: 
# "Back of the envelope" wind and altitude correction for 100 metre sprint times, 22.6.2000

def time_correction(time, wind, altitude):
    return time * (1.027 - 0.027 * np.exp(-0.000125 * altitude) * (1 - ((wind * time) / 100))**2)

# Convert the times to float

# Pseudocode:

# Check result column, if it contains h then remove the h, convert to float and add 0.24
# otherwise if it contains a or A, remove the a or A, and convert to float

# Removing altitude time indicators and + sign for times on route to 150m or 200m
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "a$",value ="",regex=True)
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "A$",value ="",regex=True)
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "\\+$",value ="",regex=True)

# Make an indicator where it's a 1 if there's a handtime, then for rows with a 1, remove the h

editingSprintFrame["removeHandTime"] = editingSprintFrame["result"].str.contains("h")
editingSprintFrame.loc[(editingSprintFrame["removeHandTime"] == True), "result"] = editingSprintFrame["result"].replace(to_replace = "h$", value = "", regex = True)


# Modifying rows with issues in the results as the times are completely messed up

editingSprintFrame.loc[14671, "result"] = "10.20"
editingSprintFrame.loc[16390, "result"] = "10.61"
editingSprintFrame.loc[65407, "result"] = "10.46"

# Convert results to float and add 0.24 for the hand times

editingSprintFrame["result"] = pd.to_numeric(editingSprintFrame["result"], downcast = "float")
editingSprintFrame.loc[(editingSprintFrame["removeHandTime"] == True), "result"] = editingSprintFrame["result"] + 0.24


# Rename the columns in order to make a nice dictionary
altitudes2Add = altitudes2Add.rename(columns={"Unnamed: 0": "venue", "0": "elevation"})

altitudes2Fix = dict(zip(altitudes2Add["venue"], altitudes2Add["elevation"]))

# Create a new column which is the result of mapping the series to the dictionary keys
editingSprintFrame["altitude"] = editingSprintFrame["venue"].map(altitudes2Fix)

# Fix outlier at index 79104 - data entry error, correct wind reading on World Athletics -> -192 m/s to -1.3 m/s

#editingSprintFrame.loc[79104, "wind"] = -1.3

# Create the time corrected results column by passing in the result column for time, wind for wind, and altitude
# for altitude
editingSprintFrame["time-corrected"] = time_correction(editingSprintFrame["result"], editingSprintFrame["wind"], editingSprintFrame["altitude"])


# Convert the date column and date of birth columns into date formats

editingSprintFrame["date"] = pd.to_datetime(editingSprintFrame["date"])

 # + '20:00:00'

editingSprintFrame["dateOfBirth"] = pd.to_datetime(editingSprintFrame["dateOfBirth"])
# Personal Best Column & Personal Best Column Time Corrected

# Sorts in ascending order by name and then the date of each performance, then groups by name and finds the minimum result 
# as the dates ascend, so the minimum updates to the athlete's pb
editingSprintFrame["PersonalBest"] = editingSprintFrame.sort_values(by = ["name", "date"]).groupby("name")["result"].cummin()
editingSprintFrame["PersonalBest"] = np.nan # can't include wind=aided times for regular personal best
editingSprintFrame["PersonalBest_Corrected"] = editingSprintFrame.sort_values(by = ["name", "date"]).groupby("name")["time-corrected"].cummin()

# Season Best Column & Season Best Column Time Corrected
# Sorts in ascending order by name, the date of each performance, and the overall year, then groups by date and finds the minimum result 
# as the dates ascend, so the minimum updates to the athlete's pb

# First extract the year from each performance

editingSprintFrame["yearOfResult"] = editingSprintFrame["date"].dt.year

# Then the same code as personal best, but with the year as an additional column to sort
editingSprintFrame["SeasonBest"] = editingSprintFrame.sort_values(by = ["name", "yearOfResult", "date"]).groupby(["name", "yearOfResult"])["result"].cummin()
editingSprintFrame["SeasonBest"] = np.nan # can't include wind=aided times for regular season best
editingSprintFrame["SeasonBest_Corrected"] = editingSprintFrame.sort_values(by = ["name", "yearOfResult", "date"]).groupby(["name", "yearOfResult"])["time-corrected"].cummin()

# Create a new column for age during Performance - difference between dateOfBirth and date

editingSprintFrame["ageDuringResult"] = (editingSprintFrame["date"] - editingSprintFrame["dateOfBirth"])/np.timedelta64(1, "D")/365


# Extract month of performance for Time Series Analysis:

editingSprintFrame["monthOfResult"] = editingSprintFrame["date"].dt.month

# TODO: If two performances from an athlete occur on the same day, a heat should go before a quarterfinal and a semi final should go before a final

editingSprintFrame.to_csv("C:/Users/calde/OneDrive/Documents/sprintData/updated_clean_windaided_v2.csv")

# Combining wind legal and wind aided:

previous_cleanFrame = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/updated_clean_windlegal_v2.csv")

# drop the old corrected pb and season best columns

cleanestFrame = previous_cleanFrame.drop(['PersonalBest_Corrected', 'SeasonBest_Corrected'], axis = 1)
cleanestFrame = cleanestFrame.drop(["PB", "Unnamed: 0.1"], axis = 1)
cleanestFrame = cleanestFrame.drop(["records"], axis = 1)
# Drop the wind-legal column

editingSprintFrame = editingSprintFrame.drop(["windlegal"],axis=1)
# Combine the wind-aided and wind-legal datasets together

combinedSprintTimes = pd.concat([cleanestFrame, editingSprintFrame], join = 'outer')

# Find how much missing data exists

for each_col in range(len(combinedSprintTimes.columns)):
    print(combinedSprintTimes.columns[each_col] + ": " + f'{combinedSprintTimes.iloc[0:492806, each_col].count()}')

# Remove rows with NA birth dates

combinedSprintTimes = combinedSprintTimes.dropna(subset = "dateOfBirth")

# Re-evaluate personal best corrected and season best corrected, now that there are both wind-aided and wind legal times

# Reset indexes in order to group by properly

combinedSprintTimes = combinedSprintTimes.reset_index(drop = True)
combinedSprintTimes["PersonalBest_Corrected"] = combinedSprintTimes.sort_values(by = ["name", "date"]).groupby("name")["time-corrected"].cummin()
combinedSprintTimes["SeasonBest_Corrected"] = combinedSprintTimes.sort_values(by = ["name", "yearOfResult", "date"]).groupby(["name", "yearOfResult"])["time-corrected"].cummin()

combinedSprintTimes.to_csv("C:/Users/calde/OneDrive/Documents/sprintData/analysis_ready_performances.csv")