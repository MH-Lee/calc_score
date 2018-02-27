import pandas as pd
import os

def kospi_ab_merge():
    os.chdir("./kospi_total_ab_score/")
    kospi_total_list = os.listdir()
    len(kospi_total_list)
    cols = ['date', 'code', 'name', 'individual_possession',
       'foreign_retail_possession', 'institution_possession',
       'etc_corporate_possession', 'trust_possession', 'pension_possession',
       'total_stock_in_circulation', 'individual_height',
       'foreign_retail_height', 'institution_height', 'etc_corporate_height',
       'institution_purity', 'individual_proportion',
       'foreign_retail_proportion', 'institution_proportion',
       'etc_corporate_proportion', 'individual_tp', 'foreign_retail_tp',
       'institution_tp', 'etc_corporate_tp', 'individual_apps_tp',
       'foreign_retail_apps_tp', 'institution_apps_tp',
       'etc_corporate_apps_tp', 'individual_coef', 'individual_tvalue',
       'foreign_retail_coef', 'foreign_retail_tvalue', 'institution_coef',
       'institution_tvalue', 'etc_corporate_coef', 'etc_corporate_tvalue',
       'institution_section', 'foreign_retail_section',
       'institution_proportion_section', 'foreign_retail_proportion_section',
       'institution_tvalue_section', 'foreign_retail_tvalue_section',
       'institution_coef_section', 'foreign_retail_coef_section',
       'institution_purity_section', 'institution_score',
       'foreign_retail_score', 'absolute_score']
    dfs = pd.DataFrame(columns=cols)
    i = 0
    for filename in kospi_total_list:
        df = pd.read_csv(filename, sep=',')
        df['code'] = df['code'].apply(lambda x: str(x).zfill(6))
        dfs = dfs.append(df)
        i = i + 1
        percent = round((i / len(kospi_total_list))*100,2)
        print(filename, ' ',i,'번째', percent, "%")
    dfs = dfs[cols]
    dfs.to_csv('../data/kospi_ab_score.csv', encoding='utf-8',index = False)
