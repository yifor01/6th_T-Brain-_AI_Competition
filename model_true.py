
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
from math import sqrt, floor,log,ceil,exp
from sklearn import linear_model
matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
#import pandas_profiling 
import seaborn as sns
#from ggplot import *
from sklearn.model_selection import train_test_split
from IPython.display import display
import shap
from scipy.stats import norm, skew #for some statistics
from scipy import stats
from sklearn.model_selection import GridSearchCV  


pd.set_option("display.max_columns",40)
pd.set_option("display.max_row",20)

# load data
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv("test.csv")
df = pd.merge(df_train,df_test,how='outer')


df['town'] =  10000*df.town + 10000000*df.city
df['village'] = df.village + 10000*df.town + 10000000*df.city
df['r_city'] = df.city.map(dict(df.groupby('city').village_income_median.mean()))
df['r_town'] = df.town.map(dict(df.groupby('town').village_income_median.mean()))
df.village_income_median[df.village_income_median.isna()] = df.r_town[df.village_income_median.isna()]
df.village_income_median[df.village_income_median.isna()] = df.r_city[df.village_income_median.isna()]
del df['r_town'],df['r_city']#,df['parking_area'],df['parking_price']

df_sub = pd.read_csv("submit_test.csv")  

# Fill NA
df.isnull().sum()[df.isnull().sum()>0]/df.shape[0]
df.txn_floor = df.txn_floor.fillna(-999)

### New Feature
df['build_time'] = df.txn_dt - df.building_complete_dt
del df['building_complete_dt']

df['floor_ss'] = np.int_(df.txn_floor==-999)
df['floor_1'] = np.int_(df.txn_floor==1)*(1-df['floor_ss'])+np.int_(df.total_floor==1)*df['floor_ss']
df['floor_4'] = np.int_(df.txn_floor==4)*(1-df['floor_ss'])+np.int_(df.total_floor==4)*df['floor_ss']
df['floor_top'] = np.int_(df.txn_floor==df.total_floor)*(1-df['floor_ss'])+df['floor_ss']
df['floor_rate'] = np.maximum((df.txn_floor.values-4)//4,0)
df['txn_year'] = (df['txn_dt']-min(df['txn_dt'])) /365
df['txn_mon'] = np.ceil(((df['txn_dt']-min(df['txn_dt'])) /30)%11+1)               # dummy
df['txn_day'] = (df['txn_dt']-min(df['txn_dt'])) % 7

df['build_time_year'] = (df['build_time']-min(df['build_time'])) //365
df['build_time_mon'] = np.ceil(((df['build_time']-min(df['build_time'])) /30)%11+1)# dummy
df['build_time_day'] = (df['build_time']-min(df['build_time'])) %7
df['log_build_area'] = np.log(df.building_area)
df['log_land_area'] = np.log(df.land_area+1)

df['log_parking_area']  = np.log(df.parking_area+1)
df['log_parking_price'] = np.log(df.parking_price+1)

df['log_parking_area'] = df['log_parking_area'].fillna(-1)
df['log_parking_price'] = df['log_parking_price'].fillna(-1)


# Reference variable depend with y
df['ref_1'] = np.log(df.total_price/df.building_area)

df['city_encoding'] = df.city.map(dict(df.city.value_counts()/df.shape[0] ))
df['city_lprice_mean'] = df.city.map(dict(np.log(df.groupby('city').total_price.mean())))
df['city_lprice_median'] = df.city.map(dict(np.log(df.groupby('city').total_price.median())))
df['city_ref1_mean'] = df.city.map(dict(df.groupby('city').ref_1.mean()))
df['city_ref1_median'] = df.city.map(dict(df.groupby('city').ref_1.median()))
df['city_ref1_std'] = df.city.map(dict(df.groupby('city').ref_1.std()))

df.city_ref1_std = df.city_ref1_std.fillna(0)
df.city_lprice_mean = df.city_lprice_mean.fillna(df.city_lprice_mean.mean())
df.city_lprice_median = df.city_lprice_median.fillna(df.city_lprice_median.mean())
df.city_ref1_mean = df.city_ref1_mean.fillna(df.city_ref1_mean.mean())
df.city_ref1_median = df.city_ref1_median.fillna(df.city_ref1_median.mean())

df['town_encoding'] = df.town.map(dict(df.town.value_counts()/df.shape[0] ))
df['town_lprice_mean'] = df.town.map(dict(np.log(df.groupby('town').total_price.mean())))
df['town_lprice_median'] = df.town.map(dict(np.log(df.groupby('town').total_price.median())))
df['town_ref1_mean'] = df.town.map(dict(df.groupby('town').ref_1.mean()))
df['town_ref1_median'] = df.town.map(dict(df.groupby('town').ref_1.median()))
df['town_ref1_std'] = df.town.map(dict(df.groupby('town').ref_1.std()))
df.town_ref1_std = df.town_ref1_std.fillna(0)
df.town_lprice_mean = df.town_lprice_mean.fillna(df.town_lprice_mean.mean())
df.town_lprice_median = df.town_lprice_median.fillna(df.town_lprice_median.mean())
df.town_ref1_mean = df.town_ref1_mean.fillna(df.town_ref1_mean.mean())
df.town_ref1_median = df.town_ref1_median.fillna(df.town_ref1_median.mean())

df['village_encoding'] = df.village.map(dict(df.village.value_counts()/df.shape[0] ))
df['village_lprice_mean'] = df.village.map(dict(np.log(df.groupby('village').total_price.mean())))
df['village_lprice_median'] = df.village.map(dict(np.log(df.groupby('village').total_price.median())))
df['village_ref1_mean'] = df.village.map(dict(df.groupby('village').ref_1.mean()))
df['village_ref1_median'] = df.village.map(dict(df.groupby('village').ref_1.median()))
df['village_ref1_std'] = df.village.map(dict(df.groupby('village').ref_1.std()))
df.village_ref1_std = df.village_ref1_std.fillna(0)
df.village_lprice_mean = df.village_lprice_mean.fillna(df.village_lprice_mean.mean())
df.village_lprice_median = df.village_lprice_median.fillna(df.village_lprice_median.mean())
df.village_ref1_mean = df.village_ref1_mean.fillna(df.village_ref1_mean.mean())
df.village_ref1_median = df.village_ref1_median.fillna(df.village_ref1_median.mean())


########
df['material_encoding'] = df.building_material.map(dict(df.building_material.value_counts()/df.shape[0] ))
df['material_lprice_mean'] = df.building_material.map(dict(np.log(df.groupby('building_material').total_price.mean())))
df['material_lprice_median'] = df.building_material.map(dict(np.log(df.groupby('building_material').total_price.median())))
df['material_ref1_mean'] = df.building_material.map(dict(df.groupby('building_material').ref_1.mean()))
df['material_ref1_median'] = df.building_material.map(dict(df.groupby('building_material').ref_1.median()))
df['material_ref1_std'] = df.building_material.map(dict(df.groupby('building_material').ref_1.std()))


df['type_encoding'] = df.building_type.map(dict(df.building_type.value_counts()/df.shape[0] ))
df['type_lprice_mean'] = df.building_type.map(dict(np.log(df.groupby('building_type').total_price.mean())))
df['type_lprice_median'] = df.building_type.map(dict(np.log(df.groupby('building_type').total_price.median())))
df['type_ref1_mean'] = df.building_type.map(dict(df.groupby('building_type').ref_1.mean()))
df['type_ref1_median'] = df.building_type.map(dict(df.groupby('building_type').ref_1.median()))
df['type_ref1_std'] = df.building_type.map(dict(df.groupby('building_type').ref_1.std()))


df['use_encoding'] = df.building_use.map(dict(df.building_use.value_counts()/df.shape[0] ))
df['use_lprice_mean'] = df.building_use.map(dict(np.log(df.groupby('building_use').total_price.mean())))
df['use_lprice_median'] = df.building_use.map(dict(np.log(df.groupby('building_use').total_price.median())))
df['use_ref1_mean'] = df.building_use.map(dict(df.groupby('building_use').ref_1.mean()))
df['use_ref1_median'] = df.building_use.map(dict(df.groupby('building_use').ref_1.median()))
df['use_ref1_std'] = df.building_use.map(dict(df.groupby('building_use').ref_1.std()))


# Only for mice data
#df['parking_per'] = np.log(df.parking_price/(df.parking_area+1) +1)

# transform
df['lon_t'] = df.lon-df.lat
df['lat_t'] = df.lat+df.lon

#sns.jointplot(x=df.lon_t,y=df.lat_t)



# new dummy
a1 = df[['I_MIN',"II_MIN","III_MIN","IV_MIN","V_MIN","VI_MIN","VII_MIN","VIII_MIN",
                     'IX_MIN',"X_MIN","XI_MIN","XII_MIN","XIII_MIN" ]].apply(lambda x :np.argmin(x),axis=1).\
map({'I_MIN':1,"II_MIN":2,"III_MIN":3,"IV_MIN":4,"V_MIN":5,"VI_MIN":6,"VII_MIN":7,"VIII_MIN":8,
     'IX_MIN':9,"X_MIN":10,"XI_MIN":11,"XII_MIN":12,"XIII_MIN":13})
a2 = pd.get_dummies(a1,drop_first=True)
a2.columns =['best_min_'+str(x) for x in range(1,len(a1.unique()))]

a3 = pd.get_dummies(df.city,drop_first=True)
a3.columns =['citycode_'+str(x) for x in range(1,len(df.city.unique()))]

a4 = pd.get_dummies(df['txn_mon'],drop_first=True)
a4.columns =['txn_mon_'+str(x) for x in range(1,len(df.txn_mon.unique()))]

a5 = pd.get_dummies(df['build_time_mon'],drop_first=True)
a5.columns =['build_mon_'+str(x) for x in range(1,len(df.build_time_mon.unique()))]

a6 = pd.get_dummies(df.building_use,drop_first=True)
a6.columns =['build_use_'+str(x) for x in range(1,len(df.building_use.unique()))]

a7 = pd.get_dummies(df['building_material'],drop_first=True)
a7.columns =['build_mat_'+str(x) for x in range(1,len(df.building_material.unique()))]

a8 = pd.get_dummies(df['building_type'],drop_first=True)
a8.columns =['build_type_'+str(x) for x in range(1,len(df.building_type.unique()))]

df = pd.concat([df,a2,a3,a4,a5,a6,a7,a8], axis=1)

del a1,a2,a3,a4,a5,a6,a7,a8



# EDA
#sns.jointplot(x=df.building_material,y=np.log(df.total_price/df.building_area),size=10)

# use np.log(df.total_price/df.building_area) 
#plt.hist(  np.log(df.total_price/df.building_area)  )
#plt.hist(  np.log(df.total_price)/df.building_area )

#sns.jointplot(x=df.city,y=np.log(df.total_price/df.building_area) )
#plt.hist( train.total_price/train.building_area  )
#np.log(train.total_price/train.building_area).describe()
#train[np.log(train.total_price/train.building_area)>16]
#sns.jointplot(x=df.I_MIN,y=np.log(df.total_price))
#sns.jointplot(x=df.XIV_MIN,y=np.log(df.total_price))

##################### model (consider log tranform)#########################################
import lightgbm as lgb
import catboost as cat
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
sns.set(rc={'figure.figsize':(7,7)})



# score function
def score(y_true,y_pred,show=1):
    a11 = (abs((y_true-y_pred)/y_true)<=0.1).mean()
    a22 = 1-(abs((y_true-y_pred)/y_true)).mean()
    if show==1: 
        sns.scatterplot(x=[x for x in range(len(y_true))], y= y_true-y_pred)
        plt.title('score residual ratio plot')
        plt.xlabel('data point')
        plt.ylabel('score residual ratio')
    return a11.round(4)*10000 + a22


feature = [x for x in df.columns if x not in ['total_price','city', 'ref_1','building_id','town','village',
                                              'lat','lon','parking_price','parking_area','building_area',
                                              'land_area','txn_mon','build_time_mon'] ]

train = df[df.index<60000]
test  = df[df.index>=60000]

df[feature].isna().sum()[df[feature].isna().sum()>0]



# cut outlier 13 points
#train = train[np.log(train.total_price/train.building_area)<17]

x_train,x_valid,y_train,y_valid = train_test_split(train[feature],
                                                   np.log(train.total_price/(train.building_area) ),
                                                   train_size=.8,random_state=88)


# LightGBM 1
baseline = 0
for a in [0.08,0.09,0.1,0.11,0.12,0.13]:
    for b in [7,8,9,10,11,12,13,14,15]:
        model_LGB = lgb.LGBMRegressor(objective='regression',num_leaves=20,max_depth=b,
                                      learning_rate=a, n_estimators=5000,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.4,
                                      min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        model_LGB.fit(x_train,y_train)
        #display(pd.DataFrame({'variable':x_train.columns, 'importance':model_LGB.feature_importances_}).\
        #        sort_values('importance',ascending=False)[0:20])
        ss = score(np.exp(y_valid+ x_valid.log_build_area), 
                   np.exp(model_LGB.predict(x_valid)+ x_valid.log_build_area ),show=0 )
        
        if ss > baseline:
            print('%1.2f'%a,',','%2.0f'%b,',','%5.2f'%ss," *") 
            baseline = ss
        else:
            print('%1.2f'%a,',','%2.0f'%b,',','%5.2f'%ss) 




# XGBoost 1
model_xgb = xgb.XGBRegressor(colsample_bytree=0.333, gamma=0.0468, 
                             learning_rate=0.05, max_depth=12, 
                             min_child_weight=1.7817, n_estimators=4000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.9,verbose=1,
                             random_state =7, nthread = -1)
model_xgb.fit(x_train,y_train)
display(pd.DataFrame({'feature':feature,'value':model_xgb.feature_importances_*100}).\
        sort_values(by='value',ascending=False)[0:20])
score(np.exp(y_valid+x_valid.log_build_area),
      np.exp(model_xgb.predict(x_valid)+x_valid.log_build_area) ) # 5478.865
#model_xgb.fit(train[feature],np.log(train.total_price/train.building_area))
#np.exp(model_xgb.predict(test[feature])+np.log(test.building_area)) 



# GBoost 1 
GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=12, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,verbose=1,
                                   loss='huber', random_state =5)
GBoost.fit(x_train,y_train)
display(pd.DataFrame({'feature':feature,'value':GBoost.feature_importances_*100}).\
        sort_values(by='value',ascending=False)[0:20])


score(np.exp(y_valid+x_valid.log_build_area),
      np.exp(GBoost1.predict(x_valid)+x_valid.log_build_area) ) #  5713.8701



# GBoost 2 (131) cost very much time
GBoost1 = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=12, max_features=131,verbose=1,
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost1.fit(x_train,y_train)
display(pd.DataFrame({'feature':feature,'value':GBoost1.feature_importances_*100}).\
        sort_values(by='value',ascending=False)[0:20])
score(np.exp(y_valid+x_valid.log_build_area),
      np.exp(GBoost1.predict(x_valid)+x_valid.log_build_area) ) #  5721.869


GBoost1.fit(train[feature],np.log(train.total_price/train.building_area))
np.exp(GBoost1.predict(test[feature])+test.log_build_area)





sns.jointplot(x=df.VIII_5000, y= df.ref_1)
sns.jointplot(x=df.VIII_10000, y= df.ref_1)

sns.jointplot(x=df.VIII_10000-df.VIII_1000, y= df.ref_1)



sns.jointplot(x=df.VIII_10000-df.VIII_100, y= df.ref_1)
sns.jointplot(x=df.XIII_10000-df.XIII_1000, y= np.log(df.total_price))








# CatBoost
CAT = cat.CatBoostRegressor(depth=12,iterations=1000,subsample=0.8,early_stopping_rounds=200,
                            learning_rate=0.037, eval_metric='MAPE',bootstrap_type='Bernoulli')
CAT.fit(x_train[feature],y_train)
score(np.exp(y_valid+np.log(x_valid.building_area)),
      np.exp(CAT.predict(x_valid)+np.log(x_valid.building_area)) ) #  


# Random Forest
from sklearn.ensemble import RandomForestRegressor as rfr
rf = rfr(n_estimators=500,max_depth=10,n_jobs=5,criterion='mae',verbose=4)
rf.fit(x_train[feature],y_train)
score(np.exp(y_valid+np.log(x_valid.building_area)),
      np.exp(rf.predict(x_valid)+np.log(x_valid.building_area)) ) #  





# log y
x_train['y1'] = (model_LGB.predict(x_train[feature])+np.log(x_train.building_area)).values
x_valid['y1']  = (model_LGB.predict(x_valid[feature])+np.log(x_valid.building_area)).values
x_train['y2'] = (model_xgb.predict(x_train[feature])+np.log(x_train.building_area)).values
x_valid['y2']  = (model_xgb.predict(x_valid[feature])+np.log(x_valid.building_area)).values
x_train['y3'] = (GBoost.predict(x_train[feature])+np.log(x_train.building_area)).values
x_valid['y3']  = (GBoost.predict(x_valid[feature])+np.log(x_valid.building_area)).values
x_train['y4'] = (ENet.predict(x_train[feature])+np.log(x_train.building_area)).values
x_valid['y4']  = (ENet.predict(x_valid[feature])+np.log(x_valid.building_area)).values
x_train['y5'] = (lasso.predict(x_train[feature])+np.log(x_train.building_area)).values
x_valid['y5']  = (lasso.predict(x_valid[feature])+np.log(x_valid.building_area)).values


# old feature

model_LGB.fit(train[feature],np.log(train.total_price/train.building_area))
print('LightGBM OK.')
model_xgb.fit(train[feature],np.log(train.total_price/train.building_area))
print('XGBoost OK.')
GBoost.fit(train[feature],np.log(train.total_price/train.building_area))
print('GBM OK.')
ENet.fit(train[feature],np.log(train.total_price/train.building_area))
print('ENet OK.')
lasso.fit(train[feature],np.log(train.total_price/train.building_area))
print('Lasso OK.')



test['y1'] = (model_LGB.predict(test[feature])+np.log(test.building_area)).values
test['y2'] = (model_xgb.predict(test[feature])+np.log(test.building_area)).values
test['y3'] = (GBoost.predict(test[feature])+np.log(test.building_area)).values
test['y4'] = (ENet.predict(test[feature])+np.log(test.building_area)).values
test['y5'] = (lasso.predict(test[feature])+np.log(test.building_area)).values



new_feature = feature.copy()
[new_feature.append(x) for x in ['y1','y2','y3','y4','y5']]

new_xtrain = pd.DataFrame({'y1':np.exp(x_train.y1),'y2':np.exp(x_train.y2),
                           'y3':np.exp(x_train.y3),'y4':np.exp(x_train.y4),'y5':x_train.y5})
    
new_xvalid = pd.DataFrame({'y1':np.exp(x_valid.y1),'y2':np.exp(x_valid.y2),
                           'y3':np.exp(x_valid.y3),'y4':np.exp(x_valid.y4),'y5':x_valid.y5})
    

################
## model 
S_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=0.0468, 
                         learning_rate=0.05, max_depth=5, 
                         min_child_weight=2, n_estimators=500,
                         reg_alpha=0.4640, reg_lambda=0.8571,
                         subsample=0.7, silent=1,
                         random_state =7, nthread = -1)

S_xgb.fit(  new_xtrain.iloc[:,1:3] ,  y_train + np.log(x_train.building_area) )
print(score( np.exp(y_valid+np.log(x_valid.building_area)), np.exp(S_xgb.predict(new_xvalid.iloc[:,1:3]) ))  ) # 5723.87
# 200 : 5742.87



S_xgb.fit(pd.concat([new_xtrain.iloc[:,1:3],new_xvalid.iloc[:,1:3]]), 
          np.hstack(((y_train + np.log(x_train.building_area)),
                     (y_valid + np.log(x_valid.building_area)))) )

# submit
np.exp(S_xgb.predict(np.exp(test.iloc[:,-4:-2])))


#############################################

# shap value
explainer = shap.TreeExplainer(model_LGB)
shap_values = explainer.shap_values(x_train)
shap_v = []
for i in range(shap_values.shape[1]):
    shap_v.append(abs(shap_values[:,i]).sum()/shap_values.shape[0])
shap_v = np.array(shap_v)
J = pd.DataFrame({"feature":x_train.columns,"shap":shap_v}).sort_values('shap',ascending=False)
m = min(len(J[J.shap!=0]), ceil(5*sqrt(x_train.shape[0] /log(3*x_train.shape[1]))))   

a = [];loss=[];early=0;sc = []
for ww in range(m):
    a.append(J.index[ww])
    lgb1 = lgb.LGBMRegressor(learning_rate=0.01,max_depth=15,n_estimators=1000,
                             silent=False,njobs=4)
    lgb1.fit(x_train.iloc[:,a],y_train,eval_set=[(x_train.iloc[:,a], y_train)],
             eval_metric=['logloss'],early_stopping_rounds=100,verbose=False)    
    tt = score(np.exp(y_valid+np.log(x_valid.building_area)),
               np.exp(lgb1.predict(x_valid.iloc[:,a])+np.log(x_valid.building_area)) ) 
    loss_ = exp( min(lgb1.evals_result_['valid_0']['l2']))*x_train.shape[0] + len(a)*2
    loss.append(loss_);sc.append(tt)
    if len(loss)>2:
        if loss[-1]>loss[-2]:
            early+=1
    if early>50:
        break
    print('#','%2.f'%(ww+1),',variable:','%3.f'%J.index[ww],',loss:','%5.2f'%loss_,',score:','%5.2f'%tt)



 
J_HDIC = J[0:np.argmin(loss)]
model_LGB.fit(x_train.iloc[:,J_HDIC.index],y_train)


score(np.exp(y_valid),np.exp(model_LGB.predict(x_valid.iloc[:,J_HDIC.index])))


# FS
a = [];loss=[];early=0
for ww in range(m):
    a.append(J.index[ww])
    lgb1 = lgb.LGBMRegressor(learning_rate=0.01,max_depth=15,n_estimators=1000,
                             silent=False,njobs=4)
    lgb1.fit(x_train.iloc[:,a],y_train,eval_set=[(x_train.iloc[:,a], y_train)],
             eval_metric=['logloss'],early_stopping_rounds=100,verbose=False)    
    
    loss_ = score(np.exp(y_valid),np.exp(lgb1.predict(x_valid.iloc[:,a])))
    if len(loss)>1:
        if max(loss)<loss_:
            early+=1
    if early==50:
        break

    loss.append(loss_)
    print('#','%3.f'%(ww+1),',','%3.f'%J.index[ww],':','%5.2f'%loss_)






dtrain = lgb.Dataset(train[feature],label = np.log(train.total_price))
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.05 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.50      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0

cv_results = lgb.cv(params, 
                    dtrain, # Using all training data
                    num_boost_round = 10000, 
                    nfold = 5, 
                    stratified = False, 
                    shuffle = True, 
                    early_stopping_rounds = 20, 
                    verbose_eval = 10, 
                    show_stdv = True, 
                    seed = 0)

# Example plot
plt.errorbar(x=range(0, len(cv_results['l1-mean'])),
             y=cv_results['l1-mean'], 
             yerr=cv_results['l1-stdv'])
plt.xlabel('Num trees')
plt.ylabel('l1-mean')
plt.show()





#OGA(x_train,np.exp(y_train))



# Catboost
'''
category_feature = []
for a in ['I',"II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV"]:
    for b in [50,500,1000,5000,10000]:
        category_feature.append(a+"_index_"+str(b))
a = ["building_material","building_type","city","building_use", 'parking_way',"topfloor","best_min"]
for i in a: category_feature.append(i)

cat_feature = np.zeros(len(category_feature),dtype='int')
for i in range(len(category_feature)):
    for j in range(len(feature)):
        if feature[j] ==category_feature[i]:
            cat_feature[i] = j
            break


model_cat = cat.CatBoostRegressor(  iterations=1000,
                                    learning_rate=0.05,
                                    depth=12,
                                    #l2_leaf_reg=None,
                                    loss_function='RMSE',
                                    thread_count=5,
                                    random_seed=None,
                                    #use_best_model=None,
                                    verbose=None,
                                    bootstrap_type='Bernoulli',
                                    subsample= 0.6,
                                    #colsample_bylevel=None,
                                    #reg_lambda=None,
                                    cat_features=cat_feature)
model_cat.fit(x_train,y_train)

model_cat.predict(x_valid)
# no log(2474)
#y1 = model_LGB.predict(x_valid)
#y1[y1<0] = min(y1[y1>0])
#score(y_valid, y1 )
# log (4578)
score(np.exp(y_valid),np.exp(model_cat.predict(x_valid)))
'''




############################# submit (check scale)##########################################
submit1 = df_sub.copy()
submit1['total_price'] = np.exp(GBoost1.predict(test[feature])+test.log_build_area).values

submit1.to_csv("test_price_s9.csv",index=False)














