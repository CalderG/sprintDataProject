import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import math
rankingData = pd.read_csv("insert ranking data csv here", index_col=0)

altitudes2Add = pd.read_csv()

# Orders the rounds by heats, quarterfinals, semifinals, and then finals last
ROUND_ORDER = {"h": 0, "q": 1, "s": 2, "f": 3}

# how many races to be averaged
WINDOW = 3

# resets indices, sorts by athleteId, year, date, and orders using the positions, then sorts them
# afterwards, it takes the results by grouping by athlete id and year and takes the previous 1, 
# and then makes a rolling average with a minimum of 1 result
# drop level is so that the rows can be reordered properly
def performance_average(rankingData, mode, window=WINDOW):
    work = rankingData.reset_index()
    work["_d"] = pd.to_datetime(work["date"], format="%m/%d/%Y")
    work["_r"] = (work["pos"].str.extract(r"\d+([a-zA-Z])")[0]
                             .map(ROUND_ORDER).fillna(9))
    work["_y"] = work["_d"].dt.year
    work = work.sort_values(["athleteId", "_y", "_d", "_r"], kind="mergesort")
    if mode == 1:
        prev = work.groupby(["athleteId", "_y"], sort=False)["result"].shift(1)
        avg = (prev.groupby([work["athleteId"], work["_y"]], sort=False)
                .rolling(window, min_periods=1).mean()
                .droplevel([0, 1]))
    else:
        prev = work.groupby(["athleteId", "_y"], sort=False)["time-corrected"].shift(1)
        avg = (prev.groupby([work["athleteId"], work["_y"]], sort=False)
                .rolling(window, min_periods=1).mean()
                .droplevel([0, 1]))
    return avg.sort_index().to_numpy()

# Placement average function - returns the average placement of the athlete's last 3 races


# Time Correction Function - credited from Jonas Mureika's paper: 
# "Back of the envelope" wind and altitude correction for 100 metre sprint times, 22.6.2000

def time_correction(time, wind, altitude):
    return time * (1.027 - 0.027 * np.exp(-0.000125 * altitude) * (1 - ((wind * time) / 100))**2)


# Position Extraction Function - extracts the position/placement of the given row, to be used for
# previous placement and other columns

def pos_extraction(position_text):
    raw_pos = re.search(r"\d+", str(position_text))
    return float(raw_pos.group()) if raw_pos else np.nan

# Takes the dataframe, resets indices and throws away old ones, formats date and time into new column,
# extracts the positions and maps them in order by round order, then sorts by athlete id and
# the new columns and merge sorts it together to then map the previous position to the current row
# sort index then restores original row order
# prev is the data frame converted to an array
def previous_placement(rankingData):
    work = rankingData.reset_index(drop=True)
    work["_d"] = pd.to_datetime(work["date"], format="%m/%d/%Y")
    work["_r"] = (work["pos"].str.extract(r"\d+([a-zA-Z])")[0]
                             .map(ROUND_ORDER).fillna(9))
    prev = (work.sort_values(["athleteId", "_d", "_r"], kind="mergesort")
                .groupby("athleteId")["pos"]
                .shift(1)
                .map(pos_extraction)
                .sort_index())
    return prev.to_numpy()
# Gets the previous result for a given athlete
def previous_result(rankingData, mode):
    work = rankingData.reset_index()
    work = work.reset_index(drop=True)
    work["_d"] = pd.to_datetime(work["date"], format="%m/%d/%Y")
    work["_r"] = (work["pos"].str.extract(r"\d+([a-zA-Z])")[0]
                             .map(ROUND_ORDER).fillna(9))
    if mode == 1:
        prev = (work.sort_values(["athleteId", "_d", "_r"], kind="mergesort")
                    .groupby("athleteId")["result"]
                    .shift(1)
                    .sort_index())
    else:
        prev = (work.sort_values(["athleteId", "_d", "_r"], kind="mergesort")
                            .groupby("athleteId")["time-corrected"]
                            .shift(1)
                            .sort_index())
    return prev.to_numpy()
# Convert the times to float

# Pseudocode:

# Check result column, if it contains h then remove the h, convert to float and add 0.24
# otherwise if it contains a or A, remove the a or A, and convert to float

# Removing altitude time indicators and + sign for times on route to 150m or 200m
rankingData["result"] = rankingData["result"].replace(to_replace = "a$",value ="",regex=True)
rankingData["result"] = rankingData["result"].replace(to_replace = "A$",value ="",regex=True)
rankingData["result"] = rankingData["result"].replace(to_replace = "\\+$",value ="",regex=True)

# Make an indicator where it's a 1 if there's a handtime, then for rows with a 1, remove the h

rankingData["removeHandTime"] = rankingData["result"].str.contains("h")
rankingData.loc[(rankingData["removeHandTime"] == True), "result"] = rankingData["result"].replace(to_replace = "h$", value = "", regex = True)


# Convert results to float and add 0.24 for the hand times

rankingData["result"] = pd.to_numeric(rankingData["result"], downcast = "float")
rankingData.loc[(rankingData["removeHandTime"] == True), "result"] = rankingData["result"] + 0.24


# Rename the columns in order to make a nice dictionary
altitudes2Add = altitudes2Add.rename(columns={"Unnamed: 0": "venue", "0": "elevation"})

altitudes2Fix = dict(zip(altitudes2Add["venue"], altitudes2Add["elevation"]))

# Create a new column which is the result of mapping the series to the dictionary keys
rankingData["altitude"] = rankingData["venue"].map(altitudes2Fix)

# Fix outlier at index 79104 - data entry error, correct wind reading on World Athletics -> -192 m/s to -1.3 m/s

#editingSprintFrame.loc[79104, "wind"] = -1.3

# Create the time corrected results column by passing in the result column for time, wind for wind, and altitude
# for altitude
rankingData["time-corrected"] = time_correction(rankingData["result"], rankingData["wind"], rankingData["altitude"])


# Convert the date column and date of birth columns into date formats

rankingData["date"] = pd.to_datetime(rankingData["date"])

rankingData["dateOfBirth"] = pd.to_datetime(rankingData["dateOfBirth"])
# Personal Best Column & Personal Best Column Time Corrected

rankingData_sorted = rankingData.sort_values(by = ["athleteId", "name", "yearOfResult", "date"])
# Sorts in ascending order by name and then the date of each performance, then groups by name and finds the minimum result 
# as the dates ascend, so the minimum updates to the athlete's pb
def season_best(x):
    vals = np.where(x["wind"] <= 2.0, x["result"].where(x["wind"] <= 2.0).cummin(), x["result"].where(x["wind"] <= 2.0).cummin().ffill())
    return pd.Series(vals, index=x.index)

rankingData_sorted["SeasonBest"] = (
    rankingData_sorted.groupby(["athleteId", "name", "yearOfResult"], group_keys=False)
    .apply(season_best, include_groups=False)
)

rankingData_sorted["PersonalBest"] = (
    rankingData_sorted.groupby(["athleteId", "name"], group_keys=False)
    .apply(season_best, include_groups=False)
)
rankingData = rankingData_sorted

rankingData["PersonalBest_Corrected"] = rankingData.sort_values(by = ["athleteId", "name", "date"]).groupby(["athleteId","name"])["time-corrected"].cummin()

# Season Best Column & Season Best Column Time Corrected
# Sorts in ascending order by name, the date of each performance, and the overall year, then groups by date and finds the minimum result 
# as the dates ascend, so the minimum updates to the athlete's pb

# First extract the year from each performance

rankingData["yearOfResult"] = rankingData["date"].dt.year

# Then the same code as personal best, but with the year as an additional column to sort

rankingData["SeasonBest_Corrected"] = rankingData.sort_values(by = ["athleteId", "name", "yearOfResult", "date"]).groupby(["athleteId", "name", "yearOfResult"])["time-corrected"].cummin()

# Create a new column for age during Performance - difference between dateOfBirth and date

rankingData["ageDuringResult"] = (rankingData["date"] - rankingData["dateOfBirth"])/np.timedelta64(1, "D")/365


# Extract month of performance for Time Series Analysis:

rankingData["monthOfResult"] = rankingData["date"].dt.month

# TODO: If two performances from an athlete occur on the same day, a heat should go before a quarterfinal and a semi final should go before a final

rankingData_sorted = rankingData.sort_values(by = "Round", 
                           key= lambda col: pd.Categorical(col.str[0], categories = ROUND_ORDER, ordered = True))

# Combining wind legal and wind aided:


# drop the old corrected pb and season best columns

#cleanestFrame = previous_cleanFrame.drop(['PersonalBest_Corrected', 'SeasonBest_Corrected'], axis = 1)
#cleanestFrame = cleanestFrame.drop(["PB", "Unnamed: 0.1"], axis = 1)
#cleanestFrame = cleanestFrame.drop(["records"], axis = 1)
# Drop the wind-legal column

#editingSprintFrame = editingSprintFrame.drop(["windlegal"],axis=1)
# Combine the wind-aided and wind-legal datasets together

#combinedSprintTimes = pd.concat([cleanestFrame, editingSprintFrame], join = 'outer')

# Find how much missing data exists

#for each_col in range(len(combinedSprintTimes.columns)):
    #print(combinedSprintTimes.columns[each_col] + ": " + f'{combinedSprintTimes.iloc[0:492806, each_col].count()}')

# Remove rows with NA birth dates

#combinedSprintTimes = combinedSprintTimes.dropna(subset = "dateOfBirth")

# Re-evaluate personal best corrected and season best corrected, now that there are both wind-aided and wind legal times

# Reset indexes in order to group by properly

rankingData = rankingData.reset_index(drop = True)
rankingData["PersonalBest_Corrected"] = rankingData.sort_values(by = ["athleteId", "name", "date"]).groupby(["athleteId","name"])["time-corrected"].cummin()
rankingData["SeasonBest_Corrected"] = rankingData.sort_values(by = ["athleteId","name", "yearOfResult", "date"]).groupby(["athleteId", "name", "yearOfResult"])["time-corrected"].cummin()

# Previous Placement Column - takes the placement from a previous race for each athlete
rankingData["PreviousPlacement"] = previous_placement(rankingData)
# Previous Result -  takes the result from a previous race for each athlete
rankingData["PreviousResult"] = previous_result(rankingData, 1)
# Previous Time-Corrected - takes the result from a previous race for each athlete, but time corrected
rankingData["PreviousTime-Corrected"] = previous_result(rankingData, 2)
# Performance Moving Average - average of last 3 results per athlete (per year) or fewer results for earlier in the season 
rankingData["PerformanceMovingAverage"] = performance_average(rankingData, 1)
# Performance Average (Time-Corrected) - average of last 3 time-corrected results per athlete (per year) or fewer results for earlier in the season
rankingData["PerformanceMovingAverage_Time-Corrected"] = performance_average(rankingData, 2)
# Placement Average - average of last 3 (or fewer if lack of data) placements in races
rankingData["PlacementAverage"] = rankingData.groupby(["athleteId", 'date'], sort=False)["PreviousPlacement"].rolling(WINDOW, min_periods=1).mean().droplevel([0, 1]).sort_index().to_numpy()
# Sub-10 Rate (Time-Corrected & Regular Legal) - gpt5-mini-assisted
# First sorts by id, name, and date, then creates a new column based on both statements being true and converts to integer (1,0)
# Then does the groupby and expands on the boolean column in order to get the rates
# Create indicator for wind-legal sub-10 performances and compute expanding mean per athlete-season
rankingData = rankingData.sort_values(by=["athleteId", "name", "date"])  # ensure chronological order per athlete
rankingData["is_sub10_windlegal"] = ((rankingData["result"] < 10) & (rankingData["wind"] <= 2.0)).astype(int)
given_athlete = rankingData.groupby("athleteId")["is_sub10_windlegal"]
rankingData["Sub10Rate-Regular"] = given_athlete.cumsum() / (given_athlete.cumcount() + 1)
# Sub-10 Time-Corrected - similar to sub 10 wind legal but wind no longer matters as it's time corrected
rankingData["is_sub10_time_corrected"] = ((rankingData["time-corrected"] < 10)).astype(int)
given_athlete_time_corrected = rankingData.groupby("athleteId")["is_sub10_time_corrected"]
rankingData["Sub10Rate-Time-Corrected"] = given_athlete_time_corrected.cumsum() / (given_athlete_time_corrected.cumcount() + 1)
# Same columns but seasonal as well

# Time Corrected and Seasonal
given_athlete_time_corrected = rankingData.groupby(["athleteId", 'yearOfResult'])["is_sub10_time_corrected"]
rankingData["Sub10Rate-Time-Corrected-Seasonal"] = given_athlete_time_corrected.cumsum() / (given_athlete_time_corrected.cumcount() + 1)
# Regular and Seasonal
given_athlete = rankingData.groupby(["athleteId", "yearOfResult"])["is_sub10_windlegal"]
rankingData["Sub10Rate-Regular-Seasonal"] = given_athlete.cumsum() / (given_athlete.cumcount() + 1)
# Get result column back after dropping it with droplevel([0,1])
rankingData = rankingData.reset_index()