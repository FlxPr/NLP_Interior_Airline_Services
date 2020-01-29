import os
import pandas as pd
import csv

a = pd.read_csv('Scraped/scrapedEGYPTAIR.csv', index_col=0)
all_data = pd.DataFrame(columns=a.columns)
all_data.to_csv('tripadvisor_all_data.csv')

for file in os.listdir("/Users/michaelullah/Documents/DSBA/CS/S2/SuperCase-elevenstrategy/"
                       "NLP_Interior_Airline_Services/Scraped"):
    if file.startswith("scraped"):
        print(os.path.join("/Scraped", file))
        df = pd.read_csv(os.path.join("Scraped", file), index_col=0)
        df.to_csv('tripadvisor_all_data.csv', mode='a', header=False)


df = pd.read_csv('tripadvisor_all_data.csv', index_col=0)
df.drop_duplicates(inplace=True)


df['star'] = df['star'].str[-2:-1]
df['date'] = df['date'].str.split('wrote a review').str[1]
df['departure'] = df['route'].str.split(' - ').str[0]
df['arrival'] = df['route'].str.split(' - ').str[0]
df['Legroom'] = df['Legroom'].str[-2:-1]
df['Seat comfort'] = df['Seat comfort'].str[-2:-1]
df['In-flight Entertainment'] = df['In-flight Entertainment'].str[-2:-1]
df['Customer service'] = df['Customer service'].str[-2:-1]
df['Value for money'] = df['Value for money'].str[-2:-1]
df['Check-in and boarding'] = df['Check-in and boarding'].str[-2:-1]
df['Food and Beverage'] = df['Food and Beverage'].str[-2:-1]
df['Cleanliness'] = df['Cleanliness'].str[-2:-1]


df.to_csv('tripadvisor_all_clean.csv', index=False)
all_clean = pd.read_csv('tripadvisor_all_clean.csv')

