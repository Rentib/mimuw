#import "@preview/touying:0.6.1": *
#import themes.university: *

#show: university-theme.with(
  config-info(
    title: [Warsaw Stock Exchange Analysis],
    author: [Stanisław Bitner],
    date: datetime.today(),
    institution: [University of Warsaw, Faculty of Mathematics, Informatics and Mechanics],
  ),
)

= Warsaw Stock Exchange Analysis

== Introduction

#pause
- *Project Overview*: Analyzing GPW financial data using a three-layered approach (Bronze, Silver, Gold).
#pause
- *Data Source*: Warsaw Stock Exchange (GPW) archive.
#pause
- *Why Chosen*: 
  #pause
  - Comprehensive dataset of stocks and bonds.
  #pause
  - Publicly available and relevant for financial market insights.

#image("./assets/logo_GPW_pl.png")

== Data Model

#pause
- *Bronze Layer*: 
  #pause
  - Raw data (e.g., Date, Name, ISIN, Open, Close, Volume).
  #pause
  - Tagged as "stock" or "bond".
#pause
- *Silver Layer*: 
  #pause
  - Cleaned data with features like DailyReturn, IntradayVolatility, MA5 (moving average).
#pause
- *Gold Layer*: 
  #pause
  - Insights such as volatility, trading volume, and correlations.

== Data Ingestion (Bronze)

#pause
- *Source*: `xls` files from GPW archive (1995-2025) scraped and stored in `csv` format.
#pause
- *Method*: 
  #pause
  - Loaded using PySpark.
  #pause
  - Changed column names to english (e.g., "Kurs_zamkniecia" to "Close").
  #pause
  - Parsed dates and saved in `parquet` format.

== Data Transformation (Silver)

#pause
- *Cleaning*: 
  #pause
  - Filled null values (e.g., Currency = "PLN").
  #pause
  - Filtered invalid records (e.g., Close > 0, Volume > 0).
#pause
- *Feature Engineering*: 
  #pause
  - DailyReturn: `(Close - PrevClose) / PrevClose`.
  #pause
  - IntradayVolatility: `(High - Low) / Open`.
  #pause
  - Moving Average (MA5), LiquidityScore, SizeBucket.

#pause

*Output*: Enriched dataset saved in `parquet` format.

== Insights and Analysis (Gold)

#pause
- *Top Volatile Stocks*: High return volatility (e.g., INTAKUS, SKYSTONE).
#pause
- *Top Volume Stocks*: High trading activity (e.g., BIOTON, PGNIG).
#pause
- *Bond-Stock Correlation*: Negative correlation (-0.1191) between DS bonds and stocks.

== Top Volatile Stocks

#image("./assets/top_volatile_stocks.png")

== Top Volume Stocks

#image("./assets/top_volume_stocks.png")

== Bond-Stock Correlation

#image("./assets/bond_stock_correlation.png")

== Conclusion

- *Key Findings*:
  - High volatility in stocks like INTAKUS suggests risk/opportunity.
  - High-volume stocks (e.g., BIOTON) indicate strong market interest.
  - Negative bond-stock correlation offers diversification potential.
- *Future Work*:
  - Incorporate dividend data.
  - Explore predictive models using machine learning.

== References

#show link: underline
- #link("https://www.gpw.pl/archiwum-notowan")[GPW Data Archive]
- #link("https://spark.apache.org/docs/latest/api/python/user_guide/index.html")[PySpark Documentation]
