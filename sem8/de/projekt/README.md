# Date Engineering -- Project

A project is in a form of a jupyter notebook with analysis of GPW data.

## Getting Data

To get the data for the project, one may use scraper `./scraper/scraper.py`.
To get stocks data, the type should be set to 10, for bonds it should be 13.
The data should be put in the `./data` directory with the following structure:
```
data
├── gpw_bonds.csv
└── gpw_stocks.csv

```
An alternative way to get the data is to get it like this:
```
wget https://students.mimuw.edu.pl/~sb438247/de_gpw_data.zip
unzip de_gpw_data.zip
```

## Docker

It should be sufficient to run the following command.

```
docker run -p 8888:8888 -p 4040:4040 --name pyspark-notebook -v $(pwd):/home/jovyan/work jupyter/all-spark-notebook:latest
```

Afterward one may open the notebook `./report.ipynb`.
