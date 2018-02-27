import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import time
import os

os.chdir('./etf_buysell')
etf = os.listdir()
os.chdir('..')

os.chdir("./etf_as")
done_list = os.listdir()
os.chdir("..")

ds = set(done_list)
etf = [x for x in etf if x not in ds]
print(len(etf))

def etf_absolute_score():
    t = 0
    for filename in etf:
        t = t + 1
        percent = round(t/len(etf),2)*100
        print(filename, percent, '%')
        etf_defacto = pd.read_csv('./etf_net/' + filename, sep=',')
        etf_ohlcv = pd.read_csv('./etf_ohlcv/' + filename, sep=',', encoding='CP949')
        etf_buy = pd.read_csv('./etf_buy/' + filename, sep=',')
        etf_defacto['code'] = etf_defacto['code'].apply(lambda x: str(x).zfill(6))
        etf_defacto = etf_defacto.set_index(['date','code'])
        etf_buy['code'] = etf_buy['code'].apply(lambda x: str(x).zfill(6))
        etf_buy = etf_buy.set_index(['date','code'])
        labels = ['individual','foreign_retail','institution','etc_corporate']
        etf_buy = etf_buy[labels]
        etf_buy.columns = ['individual_buy','foreign_retail_buy','institution_buy','etc_corporate_buy']
        etf_ohlcv['code'] = etf_ohlcv['code'].apply(lambda x: str(x).zfill(6))
        etf_ohlcv= etf_ohlcv[(etf_ohlcv['date'] > 20060101) & (etf_ohlcv['date'] < 20180215)]
        etf_ohlcv= etf_ohlcv.set_index(['date','code'])
        etf_ohlcv.drop(['name','close_price'], axis=1, inplace=True)
        etf_defacto = pd.concat([etf_defacto, etf_ohlcv, etf_buy],axis=1)
        etf_defacto = etf_defacto.reset_index('code')
        etf_defacto.index = pd.to_datetime(etf_defacto.index, format='%Y%m%d')

        # calculate possession & total total_stock_in_circulation
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            etf_defacto[agent+'_possession'] = etf_defacto[agent].cumsum() + abs(min(etf_defacto[agent].cumsum()))
        etf_defacto['total_stock_in_circulation'] = etf_defacto['individual_possession'] + etf_defacto['foreign_retail_possession'] + etf_defacto['institution_possession'] + etf_defacto['etc_corporate_possession']

        # remove zero in each agent_possession
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            etf_defacto[agent+'_possession'] = [1 if x==0 else x for x in etf_defacto[agent+'_possession']]
        etf_defacto['total_stock_in_circulation'] = [1 if x==0 else x for x in etf_defacto['total_stock_in_circulation']]

        # calculate height
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            etf_defacto[agent+'_height'] = round(etf_defacto[agent+'_possession']/etf_defacto['total_stock_in_circulation'],3)
        etf_defacto['institution_purity'] = round((etf_defacto['trust_possession'] + etf_defacto['pension_possession'])/etf_defacto['total_stock_in_circulation'],3)

        # calculate proportion
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            etf_defacto[agent+'_proportion'] = round(etf_defacto[agent+'_buy']/etf_defacto['volume'],3)

        # calculate average_price
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            etf_defacto[agent + '_tp'] = 0
            etf_defacto.loc[(etf_defacto[agent] > 0) & (etf_defacto['close_price'] > etf_defacto['open_price']), agent + '_tp'] = (etf_defacto[agent+'_height']*(((3*etf_defacto['low_price'])+etf_defacto['high_price'])/4))+((1-etf_defacto[agent+'_height'])*(((3*etf_defacto['high_price'])+etf_defacto['low_price'])/4))
            etf_defacto.loc[(etf_defacto[agent] > 0) & (etf_defacto['close_price'] == etf_defacto['open_price']), agent + '_tp'] = (etf_defacto['high_price']+etf_defacto['low_price'])/2
            etf_defacto.loc[(etf_defacto[agent] > 0) & (etf_defacto['close_price'] < etf_defacto['open_price']), agent + '_tp'] = (etf_defacto[agent+'_height']*(((3*etf_defacto['low_price'])+etf_defacto['high_price'])/4))+((1-etf_defacto[agent+'_height'])*(((3*etf_defacto['high_price'])+etf_defacto['low_price'])/4))

            etf_defacto.loc[(etf_defacto[agent] == 0) & (etf_defacto['close_price'] > etf_defacto['open_price']), agent + '_tp'] = (etf_defacto['high_price']+etf_defacto['low_price'])/2
            etf_defacto.loc[(etf_defacto[agent] == 0) & (etf_defacto['close_price'] == etf_defacto['open_price']), agent + '_tp'] = (etf_defacto['high_price']+etf_defacto['low_price'])/2
            etf_defacto.loc[(etf_defacto[agent] == 0) & (etf_defacto['close_price'] < etf_defacto['open_price']), agent + '_tp'] = (etf_defacto['high_price']+etf_defacto['low_price'])/2

            etf_defacto.loc[(etf_defacto[agent] < 0) & (etf_defacto['close_price'] > etf_defacto['open_price']), agent + '_tp'] = (etf_defacto[agent+'_height']*(((3*etf_defacto['high_price'])+etf_defacto['low_price'])/4))+((1-etf_defacto[agent+'_height'])*(((3*etf_defacto['low_price'])+etf_defacto['high_price'])/4))
            etf_defacto.loc[(etf_defacto[agent] < 0) & (etf_defacto['close_price'] == etf_defacto['open_price']), agent + '_tp'] = (etf_defacto['high_price']+etf_defacto['low_price'])/2
            etf_defacto.loc[(etf_defacto[agent] < 0) & (etf_defacto['close_price'] < etf_defacto['open_price']), agent + '_tp'] = (etf_defacto[agent+'_height']*(((3*etf_defacto['high_price'])+etf_defacto['low_price'])/4))+((1-etf_defacto[agent+'_height'])*(((3*etf_defacto['low_price'])+etf_defacto['high_price'])/4))

        drop_list = ['individual', 'foreign_retail', 'institution', 'trust', 'pension', 'etc_corporate', 'open_price', 'high_price', 'low_price', 'volume']
        etf_defacto.drop(drop_list, axis=1, inplace=True)
        # average price per share of net amount with net true price
        drop_list = ['individual_buy', 'foreign_retail_buy', 'institution_buy', 'etc_corporate_buy']
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            etf_defacto[agent + '_n_p_cumsum'] = etf_defacto[agent + '_buy'].cumsum()
            etf_defacto[agent + '_tp*n_p'] = etf_defacto[agent + '_buy']*etf_defacto[agent+'_tp']
            etf_defacto[agent + '_tp*n_p_cumsum'] = etf_defacto[agent + '_tp*n_p'].cumsum()
            etf_defacto[agent + '_apps_tp'] = round(etf_defacto[agent + '_tp*n_p_cumsum']/etf_defacto[agent + '_n_p_cumsum'],2)
            tmp_list = [agent+'_n_p_cumsum', agent + '_tp*n_p', agent + '_tp*n_p_cumsum']
            drop_list = drop_list + tmp_list
        etf_defacto.drop(drop_list, axis=1, inplace=True)
        etf_defacto.fillna(0, inplace=True)

        # statistical significance
        date_list = sorted(list(set(etf_defacto.index.strftime('%Y-%m'))))
        statistic_list = [[date_list[0],0,0,0,0,0,0,0,0]]
        j = 0
        for i in range(len(date_list[:-1])):
            if i < 12:
                j = j
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=etf_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
            else:
                j = j+1
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=etf_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
        cols=['date','individual_coef','foreign_retail_coef','institution_coef','etc_corporate_coef', 'individual_tvalue', 'foreign_retail_tvalue', 'institution_tvalue', 'etc_corporate_tvalue']
        statistical_df = pd.DataFrame(statistic_list, columns=cols)
        statistical_df=statistical_df.set_index('date')

        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            etf_defacto[agent + '_coef'] = 0
            etf_defacto[agent + '_tvalue'] = 0
            for date in statistical_df.index:
                etf_defacto.loc[(etf_defacto.index.strftime('%Y-%m') == date),agent + '_coef'] = [statistical_df.loc[date][agent + '_coef']]*etf_defacto[date].shape[0]
                etf_defacto.loc[(etf_defacto.index.strftime('%Y-%m') == date),agent + '_tvalue'] = [statistical_df.loc[date][agent + '_tvalue']]*etf_defacto[date].shape[0]

        etf_defacto['individual_tvalue'] = [3 if x>=3 else x for x in etf_defacto['individual_tvalue']]
        etf_defacto['institution_tvalue'] = [3 if x>=3 else x for x in etf_defacto['institution_tvalue']]
        etf_defacto['foreign_retail_tvalue'] = [3 if x>=3 else x for x in etf_defacto['foreign_retail_tvalue']]
        etf_defacto['etc_corporate_tvalue'] = [3 if x>=3 else x for x in etf_defacto['etc_corporate_tvalue']]
        etf_defacto.drop('close_price', axis=1, inplace=True)

        ih_Q0 = np.percentile(etf_defacto['institution_height'], 0)
        ih_Q1 = np.percentile(etf_defacto['institution_height'], 25)
        ih_Q2 = np.percentile(etf_defacto['institution_height'], 50)
        ih_Q3 = np.percentile(etf_defacto['institution_height'], 75)
        ip_Q0 = np.percentile(etf_defacto['institution_purity'], 0)
        ip_Q1 = np.percentile(etf_defacto['institution_purity'], 25)
        ip_Q2 = np.percentile(etf_defacto['institution_purity'], 50)
        ip_Q3 = np.percentile(etf_defacto['institution_purity'], 75)
        ic_Q0 = np.percentile(etf_defacto['institution_coef'], 0)
        ic_Q1 = np.percentile(etf_defacto['institution_coef'], 25)
        ic_Q2 = np.percentile(etf_defacto['institution_coef'], 50)
        ic_Q3 = np.percentile(etf_defacto['institution_coef'], 75)
        itv_Q0 = np.percentile(etf_defacto['institution_tvalue'], 0)
        itv_Q1 = np.percentile(etf_defacto['institution_tvalue'], 25)
        itv_Q2 = np.percentile(etf_defacto['institution_tvalue'], 50)
        itv_Q3 = np.percentile(etf_defacto['institution_tvalue'], 75)

        fh_Q0 = np.percentile(etf_defacto['foreign_retail_height'], 0)
        fh_Q1 = np.percentile(etf_defacto['foreign_retail_height'], 25)
        fh_Q2 = np.percentile(etf_defacto['foreign_retail_height'], 50)
        fh_Q3 = np.percentile(etf_defacto['foreign_retail_height'], 75)
        fc_Q0 = np.percentile(etf_defacto['foreign_retail_coef'], 0)
        fc_Q1 = np.percentile(etf_defacto['foreign_retail_coef'], 25)
        fc_Q2 = np.percentile(etf_defacto['foreign_retail_coef'], 50)
        fc_Q3 = np.percentile(etf_defacto['foreign_retail_coef'], 75)
        ftv_Q0 = np.percentile(etf_defacto['foreign_retail_tvalue'], 0)
        ftv_Q1 = np.percentile(etf_defacto['foreign_retail_tvalue'], 25)
        ftv_Q2 = np.percentile(etf_defacto['foreign_retail_tvalue'], 50)
        ftv_Q3 = np.percentile(etf_defacto['foreign_retail_tvalue'], 75)

        etf_defacto['institution_section'] = np.where(etf_defacto['institution_height']>ih_Q3,4,np.where(etf_defacto['institution_height']>ih_Q2,3,np.where(etf_defacto['institution_height']>ih_Q1,2,1)))
        etf_defacto['foreign_retail_section'] = np.where(etf_defacto['foreign_retail_height']>ip_Q3,4,np.where(etf_defacto['foreign_retail_height']>ip_Q2,3,np.where(etf_defacto['foreign_retail_height']>ip_Q1,2,1)))
        etf_defacto['institution_proportion_section'] = np.where(etf_defacto['institution_proportion']>ih_Q3,4,np.where(etf_defacto['institution_proportion']>ih_Q2,3,np.where(etf_defacto['institution_proportion']>ih_Q1,2,1)))
        etf_defacto['foreign_retail_proportion_section'] = np.where(etf_defacto['foreign_retail_proportion']>ip_Q3,4,np.where(etf_defacto['foreign_retail_proportion']>ip_Q2,3,np.where(etf_defacto['foreign_retail_proportion']>ip_Q1,2,1)))
        etf_defacto['institution_tvalue_section'] = np.where(etf_defacto['institution_tvalue']>ih_Q3,2,np.where(etf_defacto['institution_tvalue']>ih_Q2,1.5,np.where(etf_defacto['institution_tvalue']>ih_Q1,1,0.5)))
        etf_defacto['foreign_retail_tvalue_section'] = np.where(etf_defacto['foreign_retail_tvalue']>ip_Q3,2,np.where(etf_defacto['foreign_retail_tvalue']>ip_Q2,1.5,np.where(etf_defacto['foreign_retail_tvalue']>ip_Q1,1,0.5)))
        etf_defacto['institution_coef_section'] = np.where(etf_defacto['institution_coef']>ih_Q3,2,np.where(etf_defacto['institution_coef']>ih_Q2,1.5,np.where(etf_defacto['institution_coef']>ih_Q1,1,0.5)))
        etf_defacto['foreign_retail_coef_section'] = np.where(etf_defacto['foreign_retail_coef']>ip_Q3,2,np.where(etf_defacto['foreign_retail_coef']>ip_Q2,1.5,np.where(etf_defacto['foreign_retail_coef']>ip_Q1,1,0.5)))
        etf_defacto['institution_purity_section'] = np.where(etf_defacto['institution_purity']>ip_Q3,1,np.where(etf_defacto['institution_purity']>ip_Q2,0.75,np.where(etf_defacto['institution_purity']>ip_Q1,1,0.5)))

        etf_defacto['institution_score'] = (etf_defacto['institution_section'] + etf_defacto['institution_proportion_section'] + etf_defacto['institution_coef_section'] + etf_defacto['institution_tvalue_section'])*etf_defacto['institution_purity_section']
        etf_defacto['foreign_retail_score'] = etf_defacto['foreign_retail_section'] + etf_defacto['foreign_retail_proportion_section'] + etf_defacto['foreign_retail_coef_section'] + etf_defacto['institution_tvalue_section']
        etf_defacto['absolute_score'] = etf_defacto['institution_score'] + etf_defacto['foreign_retail_score']

        etf_defacto.to_csv('./etf_as/'+filename, sep=',', encoding='utf-8')
