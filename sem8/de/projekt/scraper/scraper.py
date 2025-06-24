import requests
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

GPW_URL = "https://www.gpw.pl/archiwum-notowan"
STOCKS_TYPE = 10
BONDS_TYPE = 13

MAX_THREADS = 8
lock = threading.Lock()

# https://www.gpw.pl/archiwum-notowan?fetch=1&type=13&instrument=&date=09-06-2025

def fetch_gpw_archive(date: str) -> pd.DataFrame:
    # params = {"fetch": 1, "type": STOCKS_TYPE, "instrument": "", "date": date}
    params = {"fetch": 1, "type": BONDS_TYPE, "instrument": "", "date": date}
    try:
        response = requests.get(GPW_URL, params=params, timeout=10)
        response.raise_for_status()

        df = pd.read_excel(response.content)
        if not df.empty:
            df["Date"] = date
        return df
    except Exception as e:
        print(f"Error fetching data for {date}: {str(e)}")
        return pd.DataFrame()


def process_date(date, results):
    date_str = date.strftime("%d-%m-%Y")
    print(f"Processing {date_str}...")
    df = fetch_gpw_archive(date_str)

    if not df.empty:
        with lock:
            results.append(df)


def main():
    start_date = datetime.datetime.now() - datetime.timedelta(days=365 * 30)
    end_date = datetime.datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_date, date, results) for date in date_range]
        for future in as_completed(futures):
            pass

    if results:
        all_data = pd.concat(results, ignore_index=True)
        all_data.to_csv("gpw_archive.csv", index=False)
        print(f"Saved data with {len(all_data)} rows to gpw_archive.csv")
    else:
        print("No data was fetched")


if __name__ == "__main__":
    main()
