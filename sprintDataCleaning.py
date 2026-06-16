import pandas as pd
import matplotlib.pyplot as plt
import requests

rawSprintFrame = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/updated_windaided_and_handtiming.csv", low_memory = False)

# Check dimensions and emptiness of columns

rawSprintFrame.shape
# 156985 rows, 21 cols

for each_col in range(len(rawSprintFrame.columns)):
    print(rawSprintFrame.columns[each_col] + ": " + f'{rawSprintFrame.iloc[0:156985, each_col].count()}')

# Remove the following columns: field1, rank, secondaryCountry, 
# nameLong, and ranking for lack of data

# Records may be kept or removed later

# Axis set to 1 to specify columns
cleanerSprintFrame = rawSprintFrame.drop(['field1', 'secondaryCountry', 
'nameLong', 'ranking', "Unnamed: 0"], axis = 1)

# Remove rows with no wind readings
cleanerSprintFrame = cleanerSprintFrame.dropna(subset = "wind")

# Remove rows with no known venues

cleanerSprintFrame = cleanerSprintFrame[cleanerSprintFrame["venue"] != "UNKNOWN"]

# Remove rows with no known placements

cleanerSprintFrame = cleanerSprintFrame.dropna(subset = "pos")

# Checking the size of the dataset after cleaning

cleanerSprintFrame.shape

# 99814 rows, 16 cols
# Build dictionary of venues to find altitudes for

venuesList = list(cleanerSprintFrame.loc[0:99814, "venue"].unique())
similarVenues = pd.DataFrame(venuesList, columns = ["venue"])
venuesCheck = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/altitudes4Venues.csv")
venuesCheck = venuesCheck.rename(columns={"Unnamed: 0": "venue", "0": "elevation"})


altitudes = []

merged = pd.merge(similarVenues, venuesCheck, on="venue", how="inner")
# Pull rows that only exist in the first dataframe
venuesFinalMerge = pd.merge(similarVenues, merged, on = "venue", how = "left_anti")

venuesList = venuesFinalMerge["venue"].tolist()

## Current Altitude Game Plan: Get lat and long then input into other api to get elevation

base_Lat_Long = "https://geocode.maps.co/search?q="

# https://tessadem.com/api/elevation?key=
base_Elevation = "https://www.elevation-api.eu/v1/elevation/"

# build list of blank altitudes
for every_venue in venuesList:
    newUrl = base_Lat_Long + every_venue + "&api_key=INSERT_APIKEY_HERE"
    latLong_Response = requests.get(newUrl)
    try:
        potentialResponse = latLong_Response.json()
        givenLat = potentialResponse[0]["lat"]
        givenLong = potentialResponse[0]["lon"]
        finalUrl = base_Elevation + givenLat + "/" + givenLong
        altitudeExtraction = requests.get(finalUrl)
        altitudes.append(altitudeExtraction.json())
    except:
        print(f"No altitude found for {every_venue}")
        altitudes.append("Replace this with manually found altitude")
   

# Finding the indices where no altitude was found - quick line from Stack Overflow

venueindices = [i for i, x in enumerate(altitudes) if x == "Replace this with manually found altitude"]

# Missing altitudes (found manually through google)
missingAltitudes = [142.9512,12.0,34.0,300.0,2696.0,57.0,89.0]

# Filling all the altitudes in
starting_altitude = 0
for each_indice in venueindices:
    altitudes[each_indice] = missingAltitudes[starting_altitude]

# Make a dictionary and save the cleaner dataframe + dictionary

venueAltitudePairs = dict(zip(venuesList, altitudes))

cleanerSprintFrame.to_csv("C:/Users/calde/OneDrive/Documents/sprintData/updated_clean_windaided.csv")

venueAltitudeFrame = pd.DataFrame.from_dict([venueAltitudePairs])

testVenueFrame = venueAltitudeFrame.T

testVenueFrame = testVenueFrame.rename(columns = {0: "elevation"})
testVenueFrame["venue"] = testVenueFrame.index
testVenueFrame = testVenueFrame[['venue', 'elevation']]
finalFrameTest = venuesCheck.merge(testVenueFrame, how = "outer")
finalFrameTest.to_csv("C:/Users/calde/OneDrive/Documents/sprintData/additionalVenues.csv")

