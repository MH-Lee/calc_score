import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import time
import os

os.chdir('./kospi_buysell')
kospi = os.listdir()
os.chdir('..')

os.chdir("./kospi_as")
done_list = os.listdir()
os.chdir("..")

ds = set(done_list)
kospi= [x for x in kospi if x not in ds]
print(len(kospi))

def kospi_absolute_score():
    t = 0
    for filename in kospi:
        t = t + 1
        percent = round(t / len(kospi), 2) * 100
        print(filename, percent,'%')
        kospi_defacto = pd.read_csv('./kospi_net/' + filename, sep=',')
        kospi_ohlcv = pd.read_csv('./kospi_ohlcv/' + filename, sep=',', encoding='CP949')
        kospi_buy = pd.read_csv('./kospi_buy/' + filename, sep=',')
        kospi_defacto['code'] = kospi_defacto['code'].apply(lambda x: str(x).zfill(6))
        kospi_defacto = kospi_defacto.set_index(['date','code'])
        kospi_buy['code'] = kospi_buy['code'].apply(lambda x: str(x).zfill(6))
        kospi_buy = kospi_buy.set_index(['date','code'])
        labels = ['individual','foreign_retail','institution','etc_corporate']
        kospi_buy = kospi_buy[labels]
        kospi_buy.columns = ['individual_buy','foreign_retail_buy','institution_buy','etc_corporate_buy']
        kospi_ohlcv['code'] = kospi_ohlcv['code'].apply(lambda x: str(x).zfill(6))
        kospi_ohlcv= kospi_ohlcv[(kospi_ohlcv['date'] > 20060101) & (kospi_ohlcv['date'] < 20180215)]
        kospi_ohlcv= kospi_ohlcv.set_index(['date','code'])
        kospi_ohlcv.drop(['name','close_price'], axis=1, inplace=True)
        kospi_defacto = pd.concat([kospi_defacto, kospi_ohlcv, kospi_buy],axis=1)
        kospi_defacto = kospi_defacto.reset_index('code')
        kospi_defacto.index = pd.to_datetime(kospi_defacto.index, format='%Y%m%d')

        # calculate possession & total total_stock_in_circulation
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            kospi_defacto[agent+'_possession'] = kospi_defacto[agent].cumsum() + abs(min(kospi_defacto[agent].cumsum()))
        kospi_defacto['total_stock_in_circulation'] = kospi_defacto['individual_possession'] + kospi_defacto['foreign_retail_possession'] + kospi_defacto['institution_possession'] + kospi_defacto['etc_corporate_possession']

        # remove zero in each agent_possession
        for agent in ['individual','foreign_retail','institution','etc_corporate','trust','pension']:
            kospi_defacto[agent+'_possession'] = [1 if x==0 else x for x in kospi_defacto[agent+'_possession']]
        kospi_defacto['total_stock_in_circulation'] = [1 if x==0 else x for x in kospi_defacto['total_stock_in_circulation']]

        # calculate height
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kospi_defacto[agent+'_height'] = round(kospi_defacto[agent+'_possession']/kospi_defacto['total_stock_in_circulation'],3)
        kospi_defacto['institution_purity'] = round((kospi_defacto['trust_possession'] + kospi_defacto['pension_possession'])/kospi_defacto['total_stock_in_circulation'],3)

        # calculate proportion
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kospi_defacto[agent+'_proportion'] = round(kospi_defacto[agent+'_buy']/kospi_defacto['volume'],3)

        # calculate average_price
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kospi_defacto[agent + '_tp'] = 0
            kospi_defacto.loc[(kospi_defacto[agent] > 0) & (kospi_defacto['close_price'] > kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto[agent+'_height']*(((3*kospi_defacto['low_price'])+kospi_defacto['high_price'])/4))+((1-kospi_defacto[agent+'_height'])*(((3*kospi_defacto['high_price'])+kospi_defacto['low_price'])/4))
            kospi_defacto.loc[(kospi_defacto[agent] > 0) & (kospi_defacto['close_price'] == kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto['high_price']+kospi_defacto['low_price'])/2
            kospi_defacto.loc[(kospi_defacto[agent] > 0) & (kospi_defacto['close_price'] < kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto[agent+'_height']*(((3*kospi_defacto['low_price'])+kospi_defacto['high_price'])/4))+((1-kospi_defacto[agent+'_height'])*(((3*kospi_defacto['high_price'])+kospi_defacto['low_price'])/4))

            kospi_defacto.loc[(kospi_defacto[agent] == 0) & (kospi_defacto['close_price'] > kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto['high_price']+kospi_defacto['low_price'])/2
            kospi_defacto.loc[(kospi_defacto[agent] == 0) & (kospi_defacto['close_price'] == kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto['high_price']+kospi_defacto['low_price'])/2
            kospi_defacto.loc[(kospi_defacto[agent] == 0) & (kospi_defacto['close_price'] < kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto['high_price']+kospi_defacto['low_price'])/2

            kospi_defacto.loc[(kospi_defacto[agent] < 0) & (kospi_defacto['close_price'] > kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto[agent+'_height']*(((3*kospi_defacto['high_price'])+kospi_defacto['low_price'])/4))+((1-kospi_defacto[agent+'_height'])*(((3*kospi_defacto['low_price'])+kospi_defacto['high_price'])/4))
            kospi_defacto.loc[(kospi_defacto[agent] < 0) & (kospi_defacto['close_price'] == kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto['high_price']+kospi_defacto['low_price'])/2
            kospi_defacto.loc[(kospi_defacto[agent] < 0) & (kospi_defacto['close_price'] < kospi_defacto['open_price']), agent + '_tp'] = (kospi_defacto[agent+'_height']*(((3*kospi_defacto['high_price'])+kospi_defacto['low_price'])/4))+((1-kospi_defacto[agent+'_height'])*(((3*kospi_defacto['low_price'])+kospi_defacto['high_price'])/4))

        drop_list = ['individual', 'foreign_retail', 'institution', 'trust', 'pension', 'etc_corporate', 'open_price', 'high_price', 'low_price', 'volume']
        kospi_defacto.drop(drop_list, axis=1, inplace=True)
        # average price per share of net amount with net true price
        drop_list = ['individual_buy', 'foreign_retail_buy', 'institution_buy', 'etc_corporate_buy']
        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kospi_defacto[agent + '_n_p_cumsum'] = kospi_defacto[agent + '_buy'].cumsum()
            kospi_defacto[agent + '_tp*n_p'] = kospi_defacto[agent + '_buy']*kospi_defacto[agent+'_tp']
            kospi_defacto[agent + '_tp*n_p_cumsum'] = kospi_defacto[agent + '_tp*n_p'].cumsum()
            kospi_defacto[agent + '_apps_tp'] = round(kospi_defacto[agent + '_tp*n_p_cumsum']/kospi_defacto[agent + '_n_p_cumsum'],2)
            tmp_list = [agent+'_n_p_cumsum', agent + '_tp*n_p', agent + '_tp*n_p_cumsum']
            drop_list = drop_list + tmp_list
        kospi_defacto.drop(drop_list, axis=1, inplace=True)
        kospi_defacto.fillna(0, inplace=True)
        date_list = sorted(list(set(kospi_defacto.index.strftime('%Y-%m'))))
        statistic_list = [[date_list[0],0,0,0,0,0,0,0,0]]
        j = 0
        for i in range(len(date_list[:-1])):
            if i < 12:
                j = j
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=kospi_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
            else:
                j = j+1
                result = smf.ols(formula='close_price ~ individual_possession + institution_possession + foreign_retail_possession + etc_corporate_possession', data=kospi_defacto.loc[date_list[j]:date_list[i]]).fit()
                tmp_list = [date_list[i+1],abs(float(format(round(result.params[1], 2), '.2f'))), abs(float(format(round(result.params[2], 2), '.2f'))), abs(float(format(round(result.params[3], 2), '.2f'))),
                            abs(float(format(round(result.params[4], 2), '.2f'))), abs(float(format(round(result.tvalues[1], 2), '.2f'))), abs(float(format(round(result.tvalues[2], 2), '.2f'))),
                            abs(float(format(round(result.tvalues[3], 2), '.2f'))), abs(float(format(round(result.tvalues[4], 2), '.2f')))]
                statistic_list.append(tmp_list)
        cols=['date','individual_coef','foreign_retail_coef','institution_coef','etc_corporate_coef', 'individual_tvalue', 'foreign_retail_tvalue', 'institution_tvalue', 'etc_corporate_tvalue']
        statistical_df = pd.DataFrame(statistic_list, columns=cols)
        statistical_df=statistical_df.set_index('date')

        for agent in ['individual','foreign_retail','institution','etc_corporate']:
            kospi_defacto[agent + '_coef'] = 0
            kospi_defacto[agent + '_tvalue'] = 0
            for date in statistical_df.index:
                kospi_defacto.loc[(kospi_defacto.index.strftime('%Y-%m') == date),agent + '_coef'] = [statistical_df.loc[date][agent + '_coef']]*kospi_defacto[date].shape[0]
                kospi_defacto.loc[(kospi_defacto.index.strftime('%Y-%m') == date),agent + '_tvalue'] = [statistical_df.loc[date][agent + '_tvalue']]*kospi_defacto[date].shape[0]

        kospi_defacto['individual_tvalue'] = [3 if x>=3 else x for x in kospi_defacto['individual_tvalue']]
        kospi_defacto['institution_tvalue'] = [3 if x>=3 else x for x in kospi_defacto['institution_tvalue']]
        kospi_defacto['foreign_retail_tvalue'] = [3 if x>=3 else x for x in kospi_defacto['foreign_retail_tvalue']]
        kospi_defacto['etc_corporate_tvalue'] = [3 if x>=3 else x for x in kospi_defacto['etc_corporate_tvalue']]
        kospi_defacto.drop('close_price', axis=1, inplace=True)

        ih_Q0 = np.percentile(kospi_defacto['institution_height'], 0)
        ih_Q1 = np.percentile(kospi_defacto['institution_height'], 25)
        ih_Q2 = np.percentile(kospi_defacto['institution_height'], 50)
        ih_Q3 = np.percentile(kospi_defacto['institution_height'], 75)
        ip_Q0 = np.percentile(kospi_defacto['institution_purity'], 0)
        ip_Q1 = np.percentile(kospi_defacto['institution_purity'], 25)
        ip_Q2 = np.percentile(kospi_defacto['institution_purity'], 50)
        ip_Q3 = np.percentile(kospi_defacto['institution_purity'], 75)
        ic_Q0 = np.percentile(kospi_defacto['institution_coef'], 0)
        ic_Q1 = np.percentile(kospi_defacto['institution_coef'], 25)
        ic_Q2 = np.percentile(kospi_defacto['institution_coef'], 50)
        ic_Q3 = np.percentile(kospi_defacto['institution_coef'], 75)
        itv_Q0 = np.percentile(kospi_defacto['institution_tvalue'], 0)
        itv_Q1 = np.percentile(kospi_defacto['institution_tvalue'], 25)
        itv_Q2 = np.percentile(kospi_defacto['institution_tvalue'], 50)
        itv_Q3 = np.percentile(kospi_defacto['institution_tvalue'], 75)

        fh_Q0 = np.percentile(kospi_defacto['foreign_retail_height'], 0)
        fh_Q1 = np.percentile(kospi_defacto['foreign_retail_height'], 25)
        fh_Q2 = np.percentile(kospi_defacto['foreign_retail_height'], 50)
        fh_Q3 = np.percentile(kospi_defacto['foreign_retail_height'], 75)
        fc_Q0 = np.percentile(kospi_defacto['foreign_retail_coef'], 0)
        fc_Q1 = np.percentile(kospi_defacto['foreign_retail_coef'], 25)
        fc_Q2 = np.percentile(kospi_defacto['foreign_retail_coef'], 50)
        fc_Q3 = np.percentile(kospi_defacto['foreign_retail_coef'], 75)
        ftv_Q0 = np.percentile(kospi_defacto['foreign_retail_tvalue'], 0)
        ftv_Q1 = np.percentile(kospi_defacto['foreign_retail_tvalue'], 25)
        ftv_Q2 = np.percentile(kospi_defacto['foreign_retail_tvalue'], 50)
        ftv_Q3 = np.percentile(kospi_defacto['foreign_retail_tvalue'], 75)

        kospi_defacto['institution_section'] = np.where(kospi_defacto['institution_height']>ih_Q3,4,np.where(kospi_defacto['institution_height']>ih_Q2,3,np.where(kospi_defacto['institution_height']>ih_Q1,2,1)))
        kospi_defacto['foreign_retail_section'] = np.where(kospi_defacto['foreign_retail_height']>ip_Q3,4,np.where(kospi_defacto['foreign_retail_height']>ip_Q2,3,np.where(kospi_defacto['foreign_retail_height']>ip_Q1,2,1)))
        kospi_defacto['institution_proportion_section'] = np.where(kospi_defacto['institution_proportion']>ih_Q3,4,np.where(kospi_defacto['institution_proportion']>ih_Q2,3,np.where(kospi_defacto['institution_proportion']>ih_Q1,2,1)))
        kospi_defacto['foreign_retail_proportion_section'] = np.where(kospi_defacto['foreign_retail_proportion']>ip_Q3,4,np.where(kospi_defacto['foreign_retail_proportion']>ip_Q2,3,np.where(kospi_defacto['foreign_retail_proportion']>ip_Q1,2,1)))
        kospi_defacto['institution_tvalue_section'] = np.where(kospi_defacto['institution_tvalue']>ih_Q3,2,np.where(kospi_defacto['institution_tvalue']>ih_Q2,1.5,np.where(kospi_defacto['institution_tvalue']>ih_Q1,1,0.5)))
        kospi_defacto['foreign_retail_tvalue_section'] = np.where(kospi_defacto['foreign_retail_tvalue']>ip_Q3,2,np.where(kospi_defacto['foreign_retail_tvalue']>ip_Q2,1.5,np.where(kospi_defacto['foreign_retail_tvalue']>ip_Q1,1,0.5)))
        kospi_defacto['institution_coef_section'] = np.where(kospi_defacto['institution_coef']>ih_Q3,2,np.where(kospi_defacto['institution_coef']>ih_Q2,1.5,np.where(kospi_defacto['institution_coef']>ih_Q1,1,0.5)))
        kospi_defacto['foreign_retail_coef_section'] = np.where(kospi_defacto['foreign_retail_coef']>ip_Q3,2,np.where(kospi_defacto['foreign_retail_coef']>ip_Q2,1.5,np.where(kospi_defacto['foreign_retail_coef']>ip_Q1,1,0.5)))
        kospi_defacto['institution_purity_section'] = np.where(kospi_defacto['institution_purity']>ip_Q3,1,np.where(kospi_defacto['institution_purity']>ip_Q2,0.75,np.where(kospi_defacto['institution_purity']>ip_Q1,1,0.5)))

        kospi_defacto['institution_score'] = (kospi_defacto['institution_section'] + kospi_defacto['institution_proportion_section'] + kospi_defacto['institution_coef_section'] + kospi_defacto['institution_tvalue_section'])*kospi_defacto['institution_purity_section']
        kospi_defacto['foreign_retail_score'] = kospi_defacto['foreign_retail_section'] + kospi_defacto['foreign_retail_proportion_section'] + kospi_defacto['foreign_retail_coef_section'] + kospi_defacto['institution_tvalue_section']
        kospi_defacto['absolute_score'] = kospi_defacto['institution_score'] + kospi_defacto['foreign_retail_score']

        kospi_defacto.to_csv('./kospi_as/'+filename, sep=',', encoding='utf-8')
