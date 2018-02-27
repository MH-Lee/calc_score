import pandas as pd
import os


etf = pd.read_csv('etf_list.csv', header=None)
etf = etf[0].tolist()
etf = [str(e).zfill(6) + '.csv' for e in etf]

os.chdir("./data_kospi")
kospi = os.listdir()
se = set(etf)
sk = set(kospi)
ske = sk - se
len(ske)
kospi_list = list(ske)
kospi_list

def buy_merge():
    cols = ["date", "code", "name", "close_price", "individual", "foreign_retail", "institution", "financial", "insurance", "trust",
        "etc_finance", "bank", "pension", "private", "nation", "etc_corporate", "foreign", "buysell"]
    dfs = pd.SparseDataFrame(columns=cols)
    i = 0
    for filename in kospi_list:
        df = pd.read_csv(filename, sep=',', encoding='CP949')
        df['code'] = df['code'].apply(lambda x: str(x).zfill(6))
        df = df.drop('Unnamed: 0',1)
        df = df[df['buysell'] == 'buy']
        df = df.drop('buysell',1)
        dfs = dfs.append(df)
        i = i + 1
        percent = round((i / len(os.listdir()))*100,2)
        print(percent,'%')
    dfs = dfs[cols]
    dfs.to_csv('kospi_buy.csv', index = False)

def sell_merge():
    cols = ["date", "code", "name", "close_price", "individual", "foreign_retail", "institution", "financial", "insurance", "trust",
        "etc_finance", "bank", "pension", "private", "nation", "etc_corporate", "foreign", "buysell"]
    dfs = pd.SparseDataFrame(columns=cols)
    i = 0
    for filename in kospi_list:
        df = pd.read_csv(filename, sep=',', encoding='CP949')
        df['code'] = df['code'].apply(lambda x: str(x).zfill(6))
        df = df.drop('Unnamed: 0',1)
        df = df[df['buysell'] == 'sell']
        df = df.drop('buysell',1)
        dfs = dfs.append(df)
        i = i + 1
        percent = round((i / len(os.listdir()))*100,2)
        print(percent,'%')
    dfs = dfs[cols]
    dfs.to_csv('kospi_sell.csv', index = False)
