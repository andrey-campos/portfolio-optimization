import os
import pandas as pd 
from dotenv import load_dotenv

# for retrieve_tickers_loaded_test to work
load_dotenv()
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")

# --- ONLY for main dev with downloaded CSV with multiple stock choices.. --
def retrieve_tickers_loaded_test() -> dict[list]:
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



# alternative version of retrieve tickers (for any dev): used to reduce overhead during development..
def retrieve_tickers_compact_test() -> dict[list]:
    return {
        "finance": ["JPM", "BAC"],
        "tech": ["AAPL", "MSFT"], 
        "industrial": ["GE", "CAT"],
        "energy": ["XOM", "CVX"]
    }


# --- In Production function to retrieve tickers from users ---
def retrieve_tickers_streamlit(uploaded_file) -> dict[list]:
    """
    ## Process uploaded CSV file from the Streamlit file_uploader

    Expected CSV format:

    `sector,ticker,company_name`

    `finance, JPM, JPMorgan Chase & Co.`

    `tech, APPL, Apple Inc`
    ...

    ### args: 
        - uploaded_file: Streamlit UploadedFile object from `st.file_uploader()`

    ### returns:
        - `dict`: Organized by each sector as a key and the values being a list of tickers.

     ### raises:
        - `ValueError`: If CSV format is invalid
        - `Exception`: For other processing errors
    """
    if uploaded_file is None: 
        raise ValueError("No File Provided")
    
    try:
        tickers_df: pd.DataFrame = pd.read_csv(uploaded_file)
        
        # validate crucial columns: sectors/tickers
        required_columns = ["sector", "ticker"]

        # will add a column and declare truthy value if it isn't in tickers_df columns
        missing_columns = [col for col in required_columns if col not in tickers_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns inside ticker data CSV: {missing_columns}")
        
        # predefine data structure to store tickers in: based off sector.
        sector_tickers = {
            "finance": [],
            "tech": [],
            "industrial": [],
            "energy": []
        }
        
        # loop through each row and clean up sector value and ticker value 
        for _, row in tickers_df.iterrows():
            sector = row["sector"].lower().strip()
            ticker = row["ticker"].upper().strip()

            if sector in sector_tickers:
                sector_tickers[sector].append(ticker)

            # if a sector doesn't exist in predefined sector_tickers dict then make a new entry 
            else:
                if sector not in sector_tickers:
                    sector_tickers[sector] = []
                sector_tickers[sector].append(ticker)

        # remove any empty sectors: will keep if v is truthy 
        sector_tickers = {k: v for k, v in sector_tickers.items() if v}

        return sector_tickers

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    
    except Exception as e:
        raise Exception(f"Unexpected error processing CSV: {str(e)}")
    
