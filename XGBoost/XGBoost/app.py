import matplotlib
from pandas.core.frame import DataFrame
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask,render_template,request
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.core import DMatrix
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,ShuffleSplit
from sklearn.metrics import mean_squared_error,r2_score
from openpyxl import load_workbook
import random
import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
@app.route("/Input_data",methods=['GET','POST'])  
def hello_world():
    global file1path,input_file,params
    if 'form1' in request.form:
        if request.method=='POST':
            file1=request.files['csvfile']
            if not os.path.isdir('static'):
                    os.mkdir('static')
            filepath=os.path.join('static',file1.filename)
            file1path=filepath
            input_file=file1.save(filepath)   
            return file1.filename
    if 'form2' in request.form:
            if request.method=='POST':
                params = request.form
            return params        
    return render_template('Input_data.html')  

@app.route("/about")
def about_world():
    return render_template('about.html') 

@app.route("/Future_data",methods=['GET','POST'])
def upload_file():
    global params,preds,test_file,file2,image1path,image2path,image3path,image4path
    if request.method=='POST':
        global file2path
        file2=request.files['csvfile']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath=os.path.join('static',file2.filename)
        file2path=filepath
        test_file=file2.save(filepath)   
        data1=pd.read_csv(file1path)
        data1 = data1.replace('',np.nan)
        data1 = data1.dropna(axis="columns", how="any")
        # data1.columns=["Sr.no","P","E","Tmin","Tmax","Q"]
        # data1['P'].round(decimals = 4)
        # data1['E'].round(decimals = 4)
        # data1['Q'].round(decimals = 4)
        # data1['Tmin'].round(decimals = 4)
        # data1['Tmax'].round(decimals = 4)
        print(data1)
        future=pd.read_csv(file2path)
        future = future.replace('',np.nan)
        future = future.dropna(axis="columns", how="any")
        # future.columns=["Sr.no","P","E","Tmin","Tmax"]
        # future.drop(['Sr.no'], axis = 1)
        # future['P'].round(decimals = 4)
        # future['E'].round(decimals = 4)
        # future['Tmin'].round(decimals = 4)
        # future['Tmax'].round(decimals = 4)
        print(future)
        data2=data1.iloc[:,-1] 
        data1.drop(data1.columns[len(data1.columns)-1], axis=1, inplace=True)
        print(data1)
        print(data2)
        data_matrix=xgb.DMatrix(data=data1,label=data2)
        data1_train, data1_test, data2_train, data2_test = train_test_split(data1, data2, test_size=float(params['test_size']))

        #XGBRegression USING Parameters entered by the user
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate =float(params["Learning_rate"]),Lambda=float(params['Lambda']),
            max_depth = int(params['Max_depth']),n_estimators = int(params['n_estimators']), subsample=float(params['subsample']),
            colsample_bytree=float(params['colsample_bytree']),alpha=float(params['alpha']),min_child_weight=int(params['min_child_weight']))
        
        a=xg_reg.fit(data1_train,data2_train)

        predict_train = a.predict(data1_train)
        predict_train=np.array(predict_train)
        predict_train=np.around(predict_train,decimals=4)
        predict_train=np.reshape(predict_train,(-1,1)) 
        rmse_train = np.sqrt(mean_squared_error(data2_train, predict_train))
        nrmse_train=rmse_train/np.average(predict_train)
        print("NRMSE_train_normal: %f" % (nrmse_train))
        train_normal=r2_score(data2_train,predict_train)
        #XGB_Test
        predict_test = a.predict(data1_test)
        predict_test=np.array(predict_test)
        predict_test=np.around(predict_test,decimals=4)
        predict_test=np.reshape(predict_test,(-1,1)) 
        rmse = np.sqrt(mean_squared_error(data2_test, predict_test))
        nrmse_test=rmse/np.average(predict_test)
        print("NRMSE_test_normal: %f" % (nrmse_test))
        test_normal=r2_score(data2_test,predict_test)

        #XGBRegression USING RandomizedSearchCV/GridSearchCV
        parameters={
        "learning_rate"    : [float(x) for x in (np.linspace(0.0, 1.0, num=21)) ] ,
        "max_depth"        : [int(x) for x in np.linspace(start = 1, stop = 15, num =16)],
        "min_child_weight" : [float(x) for x in np.linspace(start = 1, stop = 10, num = 11) ],
        "n_estimators"     : [10,100,1000],
        "subsample"        : [float(x) for x in np.linspace(start = 0.5, stop = 1, num = 11)],
        "colsample_bytree" : [float(x) for x in np.linspace(start = 0.5, stop = 1, num = 11)],
        "lambda"           : [float(x) for x in np.linspace(start = 0, stop = 5, num = 21)],
        "alpha"            : [float(x) for x in np.linspace(start = 0, stop = 5, num = 21)]
        }
        def timer(start_time=None):
            if not start_time:
                start_time = datetime.now()
                return start_time
            elif start_time:
                thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
                tmin, tsec = divmod(temp_sec, 60)
                print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

        xgb1=xgb.XGBRegressor()
        random_search=RandomizedSearchCV(estimator=xgb1,param_distributions=parameters,scoring='r2',n_jobs=-1,
        cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),verbose=2)

        start_time = timer(None) # timing starts from this point for "start_time" variable
        random_search.fit(data1,data2)
        timer(start_time) # timing ends here for "start_time" variable
       
        Best_params=random_search.best_params_
        regressor=random_search.best_estimator_

        predict_regressor_train = regressor.predict(data1_train)
        predict_regressor_train=np.array(predict_regressor_train)
        predict_regressor_train=np.around(predict_regressor_train,decimals=4)
        predict_regressor_train=np.reshape(predict_regressor_train,(-1,1)) 
        rmse = np.sqrt(mean_squared_error(data2_train, predict_regressor_train))
        nrmse_train_best=rmse/np.average(predict_regressor_train)
        print("NRMSE_train_best: %f" % (nrmse_train_best))
        train_best=r2_score(data2_train,predict_regressor_train)

        predict_regressor_test=regressor.predict(data1_test)
        predict_regressor_test=np.array(predict_regressor_test)
        predict_regressor_test=np.around(predict_regressor_test,decimals=4)
        predict_regressor_test=np.reshape(predict_regressor_test,(-1,1))
        rmse1 = np.sqrt(mean_squared_error(data2_test, predict_regressor_test))
        nrmse_test_best=rmse1/np.average(predict_regressor_test)
        print("NRMSE_tset_best: %f" % (nrmse_test_best))
        test_best=r2_score(data2_test,predict_regressor_test)

        predict_train_0 = np.array(predict_train)
        print("posNegaArr Array = ",predict_train_0)
        posArrCount = 0
        negArrCount = 0
        for i in range(len(predict_train_0)):
            if (predict_train_0[i] >= 0):
                posArrCount = posArrCount + 1
            else:
                predict_train_0[i]=0
                negArrCount = negArrCount + 1
        print("The Count of Positive Numbers in posNegaArr Array = ", posArrCount)
        print("The Count of Negative Numbers in posNegaArr Array  = ", negArrCount)

        predict_test_0 = np.array(predict_test)
        print("posNegaArr Array = ",predict_test_0)
        posArrCount = 0
        negArrCount = 0
        for i in range(len(predict_test_0)):
            if (predict_test_0[i] >= 0):
                posArrCount = posArrCount + 1
            else:
                predict_test_0[i]=0
                negArrCount = negArrCount + 1
        print("The Count of Positive Numbers in posNegaArr Array = ", posArrCount)
        print("The Count of Negative Numbers in posNegaArr Array  = ", negArrCount)

        predict_regressor_train_0= np.array(predict_regressor_train)
        print("posNegaArr Array = ",predict_test_0)
        posArrCount = 0
        negArrCount = 0
        for i in range(len(predict_regressor_train_0)):
            if (predict_regressor_train_0[i] >= 0):
                posArrCount = posArrCount + 1
            else:
                predict_regressor_train_0[i]=0
                negArrCount = negArrCount + 1
        print("The Count of Positive Numbers in posNegaArr Array = ", posArrCount)
        print("The Count of Negative Numbers in posNegaArr Array  = ", negArrCount)

        predict_regressor_test_0 = np.array(predict_regressor_test)
        print("posNegaArr Array = ",predict_regressor_test_0)
        posArrCount = 0
        negArrCount = 0
        for i in range(len(predict_regressor_test_0)):
            if (predict_regressor_test_0[i] >= 0):
                posArrCount = posArrCount + 1
            else:
                predict_regressor_test_0[i]=0
                negArrCount = negArrCount + 1
        print("The Count of Positive Numbers in posNegaArr Array = ", posArrCount)
        print("The Count of Negative Numbers in posNegaArr Array  = ", negArrCount)

        train_normal0=r2_score(data2_train,predict_train_0)
        test_normal0=r2_score(data2_test,predict_test_0)
        train_best0=r2_score(data2_train,predict_regressor_train_0)
        test_best0=r2_score(data2_test,predict_regressor_test_0)

        rmse1 = np.sqrt(mean_squared_error(data2_train, predict_train_0))
        nrmse_train_normal0=rmse1/np.average(predict_train_0)
        rmse1 = np.sqrt(mean_squared_error(data2_test, predict_test_0))
        nrmse_test_normal0=rmse1/np.average(predict_test_0)
        rmse1 = np.sqrt(mean_squared_error(data2_train, predict_regressor_train_0))
        nrmse_train_best0=rmse1/np.average(predict_regressor_train_0)
        rmse1 = np.sqrt(mean_squared_error(data2_test, predict_regressor_test_0))
        nrmse_test_best0=rmse1/np.average(predict_regressor_test_0)

        res=np.append(test_normal0,nrmse_test_normal0)
        res=np.reshape(res,(-1,1))
        res=np.around(res,decimals=4)
        df = pd.DataFrame(res,index=["R2_Score_Input_params0","NRMSE_Input_Params0"],columns=['Test'])

        res1=np.append(test_best0,nrmse_test_best0)
        res1=np.reshape(res1,(-1,1))
        res1=np.around(res1,decimals=4)
        df3 = pd.DataFrame(res1,index=["R2_Score_best0","NRMSE_best0"],columns=["Test"])
        res_test0=df.append(df3)

        res2=np.append(train_normal0,nrmse_train_normal0)
        res2=np.reshape(res2,(-1,1))
        res2=np.around(res2,decimals=4)
        df10 = pd.DataFrame(res2,index=["R2_Score_Input_params0","NRMSE_Input_Params0"],columns=['Train'])

        res3=np.append(train_best0,nrmse_train_best0)
        res3=np.reshape(res3,(-1,1))
        res3=np.around(res3,decimals=4)
        df11 = pd.DataFrame(res3,index=["R2_Score_best0","NRMSE_best0"],columns=["Train"])
        res_train0=df10.append(df11)
        result0=pd.concat([res_train0, res_test0],axis = 1)
        Datasample10=pd.DataFrame(data=result0)
        print(Datasample10)

        res=np.append(test_normal,nrmse_test)
        res=np.reshape(res,(-1,1))
        res=np.around(res,decimals=4)
        df = pd.DataFrame(res,index=["R2_Score_Input_params","NRMSE_Input_Params"],columns=['Test'])

        res1=np.append(test_best,nrmse_test_best)
        res1=np.reshape(res1,(-1,1))
        res1=np.around(res1,decimals=4)
        df3 = pd.DataFrame(res1,index=["R2_Score_best","NRMSE_best"],columns=["Test"])
        res_test=df.append(df3)

        res2=np.append(train_normal,nrmse_train)
        res2=np.reshape(res2,(-1,1))
        res2=np.around(res2,decimals=4)
        df10 = pd.DataFrame(res2,index=["R2_Score_Input_params","NRMSE_Input_Params"],columns=['Train'])

        res3=np.append(train_best,nrmse_train_best)
        res3=np.reshape(res3,(-1,1))
        res3=np.around(res3,decimals=4)
        df11 = pd.DataFrame(res3,index=["R2_Score_best","NRMSE_best"],columns=["Train"])
        res_train=df10.append(df11)

        predict_regressor_train=DataFrame(predict_regressor_train)
        predict_regressor_test=DataFrame(predict_regressor_test)
        predict_best=pd.concat([predict_regressor_train,predict_regressor_test])

        predict_regressor_train_0=DataFrame(predict_regressor_train_0)
        predict_regressor_test_0=DataFrame(predict_regressor_test_0)
        predict_best0=pd.concat([predict_regressor_train_0,predict_regressor_test_0])

        data2_train=DataFrame(data2_train)
        data2_test=DataFrame(data2_test)
        data2_best=pd.concat([data2_train,data2_test])
        Datasample9=pd.DataFrame(data2_best)
        Datasample9 = Datasample9.reset_index()

        result=pd.concat([res_train, res_test],axis = 1)

        input1_file=pd.read_csv(file1path)
        Datasample1=pd.DataFrame(data=input1_file)
        print(Datasample1)

        future_file=pd.read_csv(file2path)
        Datasample2=pd.DataFrame(data=future_file)
        print(Datasample2)
        
        params=pd.DataFrame(params,index=[0])
        Datasample3=pd.DataFrame(data=params)
        print(Datasample3)

        Best_params=pd.DataFrame(Best_params,index=[0])
        Datasample5=pd.DataFrame(data=Best_params)
        print(Datasample5)

        Datasample6=pd.DataFrame(predict_best)
        Datasample6 = Datasample6.reset_index()
        Datasample6=pd.concat([Datasample9,Datasample6],axis=1)
        print(Datasample6)

        Datasample11=pd.DataFrame(data=predict_best0)
        print(Datasample11)

        Datasample7=pd.DataFrame(data=result)
        print(Datasample7)

        future=regressor.predict(future)
        future=np.array(future)
        future=np.around(future,decimals=4)
        Datasample8=pd.DataFrame(data=future)
        print(Datasample8)
        
        with pd.ExcelWriter('output.xlsx') as writer: 
            Datasample1.to_excel(writer, sheet_name = 'Input_Data')
            Datasample3.to_excel(writer, sheet_name = 'Initial_Parameters')
            Datasample5.to_excel(writer, sheet_name = 'Optimized_Parameters')
            Datasample6.to_excel(writer, sheet_name = 'Obs(q) vs Sim(q)')
            Datasample7.to_excel(writer, sheet_name = 'Metrics')
            Datasample8.to_excel(writer, sheet_name = 'Future_Predictions') 
    return render_template('Future_data.html')

if __name__=="__main__":
    app.run(debug=True)



    