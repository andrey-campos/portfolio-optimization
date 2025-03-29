import os
import pandas as pd 
from dotenv import load_dotenv
load_dotenv()

DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")


def retrieve_tickers():
    sectors = ["finance", "tech", "industrial", "energy"]
    sectors_tickers = {}

    # Get all tickers organized by sector in a dictionary 
    for i in range(len(sectors)):
        current_sector = sectors[i] # more readability 
        csv_data = pd.read_csv(f"{DATA_FILE_PATH}/{sectors[i]}-tick.csv")

        sectors_tickers.update({current_sector: csv_data["Index Holdings and weightings as of 6:38 PM ET 11/01/2024"]})
        sectors_tickers[current_sector] = sectors_tickers[current_sector].tolist()
        del sectors_tickers[current_sector][0] # deletes the "SYMBOL" that gets imported
    
    # specific to my data
    sectors_tickers["finance"][0] = "BRK-B" # small error of BRK.b instead of BRK-B

    return sectors_tickers



