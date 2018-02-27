import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import time
import os

os.chdir('./kosdaq_buysell')
kosdaq = os.listdir()
os.chdir('..')


os.chdir("./kosdaq_as")
done_list = os.listdir()
os.chdir("..")

ds = set(done_list)
kosdaq = [x for x in etf if x not in ds]
print(len(kosdaq))

def kosdaq_absolute_score():
    for filename in kosdaq:
        kosdaq_defacto = pd.read_csv('./kosdaq_net/' + filename, sep=',')
        kosdaq_ohlcv = pd.read_csv('./kosdaq_ohlcv/' + filename, sep=',', encoding='CP949')
        kosdaq_buy = pd.read_csv('./kosdaq_buy/' + filename, sep=',')
        kosdaq_defacto['code'] = kosdaq_defacto['code'].apply(lambda x: str(x).zfill(6))
        kosdaq_defacto = kosdaq_defacto.set_index(['date','code'])
        kosdaq_buy['code'] = kosdaq_buy['code'].apply(lambda x: str(x).zfill(6))
        kosdaq_buy = kosdaq_buy.set_index(['date','code'])
        labels = ['individual','foreign_retail','institution','etc_corporate']
        kosdaq_buy = kosdaq_buy[labels]
        kosdaq_buy.columns = ['individual_buy','foreign_retail_buy','institution_buy','etc_corporate_buy']
        kosdaq_ohlcv['code'] = kosdaq_ohlcv['code'].apply(lambda x: str(x).zfill(6))
        kosdaq_ohlcv= kosdaq_ohlcv[(kosdaq_ohlcv['date'] > 20060101) & (kosdaq_ohlcv['date'] < 20180215)]
        kosdaq_ohlcv= kosdaq_ohlcv.set_index(['date','code'])
        kosdaq_ohlcv.drop(['name','close_price'], axis=1, inplace=True)
        kosdaq_defacto = pd.concat([kosdaq_defacto, kosdaq_ohlcv, kosdaq_buy],axis=1)
        kosdaq_defacto = kosdaq_defacto.reset_index('code')
        kosdaq_defacto.index = pd.to_datetime(kosdaq_defacto.index, format='%Y%m%d')

        # calculate possession & total total_stock_in_circulation
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            kosdaq_defacto[agent+'_possession'] = kosdaq_defacto[agent].cumsum() + abs(min(kosdaq_defacto[agent].cumsum()))
        kosdaq_defacto['total_stock_in_circulation'] = kosdaq_defacto['individual_possession'] + kosdaq_defacto['foreign_retail_possession'] + kosdaq_defacto['institution_possession'] + kosdaq_defacto['etc_corporate_possession']

        # remove zero in each agent_possession
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            kosdaq_defacto[agent+'_possession'] = [1 if x==0 else x for x in kosdaq_defacto[agent+'_possession']]
        kosdaq_defacto['total_stock_in_circulation'] = [1 if x==0 else x for x in kosdaq_defacto['total_stock_in_circulation']]

        # calculate height
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kosdaq_defacto[agent+'_height'] = round(kosdaq_defacto[agent+'_possession']/kosdaq_defacto['total_stock_in_circulation'],3)
        kosdaq_defacto['institution_purity'] = round((kosdaq_defacto['trust_possession'] + kosdaq_defacto['pension_possession'])/kosdaq_defacto['total_stock_in_circulation'],3)

        # calculate proportion
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kosdaq_defacto[agent+'_proportion'] = round(kosdaq_defacto[agent+'_buy']/kosdaq_defacto['volume'],3)

        # calculate average_price
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kosdaq_defacto[agent + '_tp'] = 0
            kosdaq_defacto.loc[(kosdaq_defacto[agent] > 0) & (kosdaq_defacto['close_price'] > kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto[agent+'_height']*(((3*kosdaq_defacto['low_price'])+kosdaq_defacto['high_price'])/4))+((1-kosdaq_defacto[agent+'_height'])*(((3*kosdaq_defacto['high_price'])+kosdaq_defacto['low_price'])/4))
            kosdaq_defacto.loc[(kosdaq_defacto[agent] > 0) & (kosdaq_defacto['close_price'] == kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto['high_price']+kosdaq_defacto['low_price'])/2
            kosdaq_defacto.loc[(kosdaq_defacto[agent] > 0) & (kosdaq_defacto['close_price'] < kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto[agent+'_height']*(((3*kosdaq_defacto['low_price'])+kosdaq_defacto['high_price'])/4))+((1-kosdaq_defacto[agent+'_height'])*(((3*kosdaq_defacto['high_price'])+kosdaq_defacto['low_price'])/4))

            kosdaq_defacto.loc[(kosdaq_defacto[agent] == 0) & (kosdaq_defacto['close_price'] > kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto['high_price']+kosdaq_defacto['low_price'])/2
            kosdaq_defacto.loc[(kosdaq_defacto[agent] == 0) & (kosdaq_defacto['close_price'] == kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto['high_price']+kosdaq_defacto['low_price'])/2
            kosdaq_defacto.loc[(kosdaq_defacto[agent] == 0) & (kosdaq_defacto['close_price'] < kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto['high_price']+kosdaq_defacto['low_price'])/2

            kosdaq_defacto.loc[(kosdaq_defacto[agent] < 0) & (kosdaq_defacto['close_price'] > kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto[agent+'_height']*(((3*kosdaq_defacto['high_price'])+kosdaq_defacto['low_price'])/4))+((1-kosdaq_defacto[agent+'_height'])*(((3*kosdaq_defacto['low_price'])+kosdaq_defacto['high_price'])/4))
            kosdaq_defacto.loc[(kosdaq_defacto[agent] < 0) & (kosdaq_defacto['close_price'] == kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto['high_price']+kosdaq_defacto['low_price'])/2
            kosdaq_defacto.loc[(kosdaq_defacto[agent] < 0) & (kosdaq_defacto['close_price'] < kosdaq_defacto['open_price']), agent + '_tp'] = (kosdaq_defacto[agent+'_height']*(((3*kosdaq_defacto['high_price'])+kosdaq_defacto['low_price'])/4))+((1-kosdaq_defacto[agent+'_height'])*(((3*kosdaq_defacto['low_price'])+kosdaq_defacto['high_price'])/4))

        drop_list = ['individual', 'foreign_retail', 'institution', 'trust', 'pension', 'etc_corporate', 'open_price', 'high_price', 'low_price', 'volume']
        kosdaq_defacto.drop(drop_list, axis=1, inplace=True)
        # average price per share of net amount with net true price
        drop_list = ['individual_buy', 'foreign_retail_buy', 'institution_buy', 'etc_corporate_buy']
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kosdaq_defacto[agent + '_n_p_cumsum'] = kosdaq_defacto[agent + '_buy'].cumsum()
            kosdaq_defacto[agent + '_tp*n_p'] = kosdaq_defacto[agent + '_buy']*kosdaq_defacto[agent+'_tp']
            kosdaq_defacto[agent + '_tp*n_p_cumsum'] = kosdaq_defacto[agent + '_tp*n_p'].cumsum()
            kosdaq_defacto[agent + '_apps_tp'] = round(kosdaq_defacto[agent + '_tp*n_p_cumsum']/kosdaq_defacto[agent + '_n_p_cumsum'],2)
            tmp_list = [agent+'_n_p_cumsum', agent + '_tp*n_p', agent + '_tp*n_p_cumsum']
            drop_list = drop_list + tmp_list
        kosdaq_defacto.drop(drop_list, axis=1, inplace=True)
        kosdaq_defacto.fillna(0, inplace=True)
        date_list = sorted(list(set(kosdaq_defacto.index.strftime('%Y-%m'))))

        # statistical significance
        statistic_list = [[date_list[0],0,0,0,0,0,0,0,0]]
        j = 0
        for i in range(len(date_list[:-1])):
            if i < 12:
                j = j
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=kosdaq_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
            else:
                j = j+1
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=kosdaq_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
        cols=['date','individual_coef','foreign_retail_coef','institution_coef','etc_corporate_coef', 'individual_tvalue', 'foreign_retail_tvalue', 'institution_tvalue', 'etc_corporate_tvalue']
        statistical_df = pd.DataFrame(statistic_list, columns=cols)
        statistical_df=statistical_df.set_index('date')

        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kosdaq_defacto[agent + '_coef'] = 0
            kosdaq_defacto[agent + '_tvalue'] = 0
            for date in statistical_df.index:
                kosdaq_defacto.loc[(kosdaq_defacto.index.strftime('%Y-%m') == date),agent + '_coef'] = [statistical_df.loc[date][agent + '_coef']]*kosdaq_defacto[date].shape[0]
                kosdaq_defacto.loc[(kosdaq_defacto.index.strftime('%Y-%m') == date),agent + '_tvalue'] = [statistical_df.loc[date][agent + '_tvalue']]*kosdaq_defacto[date].shape[0]

        kosdaq_defacto['individual_tvalue'] = [3 if x>=3 else x for x in kosdaq_defacto['individual_tvalue']]
        kosdaq_defacto['institution_tvalue'] = [3 if x>=3 else x for x in kosdaq_defacto['institution_tvalue']]
        kosdaq_defacto['foreign_retail_tvalue'] = [3 if x>=3 else x for x in kosdaq_defacto['foreign_retail_tvalue']]
        kosdaq_defacto['etc_corporate_tvalue'] = [3 if x>=3 else x for x in kosdaq_defacto['etc_corporate_tvalue']]
        kosdaq_defacto.drop('close_price', axis=1, inplace=True)

        ih_Q0 = np.percentile(kosdaq_defacto['institution_height'], 0)
        ih_Q1 = np.percentile(kosdaq_defacto['institution_height'], 25)
        ih_Q2 = np.percentile(kosdaq_defacto['institution_height'], 50)
        ih_Q3 = np.percentile(kosdaq_defacto['institution_height'], 75)
        ip_Q0 = np.percentile(kosdaq_defacto['institution_purity'], 0)
        ip_Q1 = np.percentile(kosdaq_defacto['institution_purity'], 25)
        ip_Q2 = np.percentile(kosdaq_defacto['institution_purity'], 50)
        ip_Q3 = np.percentile(kosdaq_defacto['institution_purity'], 75)
        ic_Q0 = np.percentile(kosdaq_defacto['institution_coef'], 0)
        ic_Q1 = np.percentile(kosdaq_defacto['institution_coef'], 25)
        ic_Q2 = np.percentile(kosdaq_defacto['institution_coef'], 50)
        ic_Q3 = np.percentile(kosdaq_defacto['institution_coef'], 75)
        itv_Q0 = np.percentile(kosdaq_defacto['institution_tvalue'], 0)
        itv_Q1 = np.percentile(kosdaq_defacto['institution_tvalue'], 25)
        itv_Q2 = np.percentile(kosdaq_defacto['institution_tvalue'], 50)
        itv_Q3 = np.percentile(kosdaq_defacto['institution_tvalue'], 75)

        fh_Q0 = np.percentile(kosdaq_defacto['foreign_retail_height'], 0)
        fh_Q1 = np.percentile(kosdaq_defacto['foreign_retail_height'], 25)
        fh_Q2 = np.percentile(kosdaq_defacto['foreign_retail_height'], 50)
        fh_Q3 = np.percentile(kosdaq_defacto['foreign_retail_height'], 75)
        fc_Q0 = np.percentile(kosdaq_defacto['foreign_retail_coef'], 0)
        fc_Q1 = np.percentile(kosdaq_defacto['foreign_retail_coef'], 25)
        fc_Q2 = np.percentile(kosdaq_defacto['foreign_retail_coef'], 50)
        fc_Q3 = np.percentile(kosdaq_defacto['foreign_retail_coef'], 75)
        ftv_Q0 = np.percentile(kosdaq_defacto['foreign_retail_tvalue'], 0)
        ftv_Q1 = np.percentile(kosdaq_defacto['foreign_retail_tvalue'], 25)
        ftv_Q2 = np.percentile(kosdaq_defacto['foreign_retail_tvalue'], 50)
        ftv_Q3 = np.percentile(kosdaq_defacto['foreign_retail_tvalue'], 75)

        kosdaq_defacto['institution_section'] = np.where(kosdaq_defacto['institution_height']>ih_Q3,4,np.where(kosdaq_defacto['institution_height']>ih_Q2,3,np.where(kosdaq_defacto['institution_height']>ih_Q1,2,1)))
        kosdaq_defacto['foreign_retail_section'] = np.where(kosdaq_defacto['foreign_retail_height']>ip_Q3,4,np.where(kosdaq_defacto['foreign_retail_height']>ip_Q2,3,np.where(kosdaq_defacto['foreign_retail_height']>ip_Q1,2,1)))
        kosdaq_defacto['institution_proportion_section'] = np.where(kosdaq_defacto['institution_proportion']>ih_Q3,4,np.where(kosdaq_defacto['institution_proportion']>ih_Q2,3,np.where(kosdaq_defacto['institution_proportion']>ih_Q1,2,1)))
        kosdaq_defacto['foreign_retail_proportion_section'] = np.where(kosdaq_defacto['foreign_retail_proportion']>ip_Q3,4,np.where(kosdaq_defacto['foreign_retail_proportion']>ip_Q2,3,np.where(kosdaq_defacto['foreign_retail_proportion']>ip_Q1,2,1)))
        kosdaq_defacto['institution_tvalue_section'] = np.where(kosdaq_defacto['institution_tvalue']>ih_Q3,2,np.where(kosdaq_defacto['institution_tvalue']>ih_Q2,1.5,np.where(kosdaq_defacto['institution_tvalue']>ih_Q1,1,0.5)))
        kosdaq_defacto['foreign_retail_tvalue_section'] = np.where(kosdaq_defacto['foreign_retail_tvalue']>ip_Q3,2,np.where(kosdaq_defacto['foreign_retail_tvalue']>ip_Q2,1.5,np.where(kosdaq_defacto['foreign_retail_tvalue']>ip_Q1,1,0.5)))
        kosdaq_defacto['institution_coef_section'] = np.where(kosdaq_defacto['institution_coef']>ih_Q3,2,np.where(kosdaq_defacto['institution_coef']>ih_Q2,1.5,np.where(kosdaq_defacto['institution_coef']>ih_Q1,1,0.5)))
        kosdaq_defacto['foreign_retail_coef_section'] = np.where(kosdaq_defacto['foreign_retail_coef']>ip_Q3,2,np.where(kosdaq_defacto['foreign_retail_coef']>ip_Q2,1.5,np.where(kosdaq_defacto['foreign_retail_coef']>ip_Q1,1,0.5)))
        kosdaq_defacto['institution_purity_section'] = np.where(kosdaq_defacto['institution_purity']>ip_Q3,1,np.where(kosdaq_defacto['institution_purity']>ip_Q2,0.75,np.where(kosdaq_defacto['institution_purity']>ip_Q1,1,0.5)))

        kosdaq_defacto['institution_score'] = (kosdaq_defacto['institution_section'] + kosdaq_defacto['institution_proportion_section'] + kosdaq_defacto['institution_coef_section'] + kosdaq_defacto['institution_tvalue_section'])*kosdaq_defacto['institution_purity_section']
        kosdaq_defacto['foreign_retail_score'] = kosdaq_defacto['foreign_retail_section'] + kosdaq_defacto['foreign_retail_proportion_section'] + kosdaq_defacto['foreign_retail_coef_section'] + kosdaq_defacto['institution_tvalue_section']
        kosdaq_defacto['absolute_score'] = kosdaq_defacto['institution_score'] + kosdaq_defacto['foreign_retail_score']

        kosdaq_defacto.to_csv('./kosdaq_as/'+filename, sep=',', encoding='utf-8')
