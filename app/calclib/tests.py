import pandas as pd
import numpy as np
import app.calclib.pipeml as pml
import app.calclib.remtime as remtime
import app.calculation as calc
import os
import remtime as rm
import json


path="C:\\Users\\avduryagin\\etc"
fpath=os.path.join(path,'features')
mpath=os.path.join(path,'models')
with open(os.path.join(path,'sid.json'),encoding='utf-8') as jsfile:
    ijson=json.load(jsfile)

width=np.array(ijson['data']['samples']['width'],dtype=np.float32)
time=np.array(ijson['data']['samples']['time'],dtype=np.float32)
s0=np.array(ijson['data']['s0'],dtype=np.float32)
s=np.array(ijson['data']['s'],dtype=np.float32)
tfirst=np.array(ijson['data']['tfirst'],dtype=np.float32)
measurements = np.array(ijson['data']['measurements'], dtype=np.float32)
t = np.array(ijson['data']['t'], dtype=np.float32)
ds = np.array(ijson['data']['sigma0'], dtype=np.float32)

#rtd=rm.rtd()
#rtd.fit(measurements=measurements,t=t,s=s,s0=s0)
#fun=rtd.predict(s_=1.1,ds=ds)
#x=np.arange(100)
#for x_ in x:
    #if np.isnan(fun(x_)):
        #print()
        #tau=fun(x_)
#properties=remtime.pipe_parameters()
#d=325.
#p=0.627
#pmax=1.6
#ch2s=0.
#cathegory=3
#kind=False
#gas_percent=0.338
#run=314.
#ryn=195.
#k=1.
#properties.fit(run=run,ryn=ryn,p=p,pmax=pmax,cathegory=cathegory,kind=kind,d=d,c_h2s=ch2s,gas_percent=gas_percent,k=k)
#val=properties.value()

#print(val)
#with open(os.path.join(path,'pyservinput_v6.json'),encoding='utf-8') as jsfile:
    #ijson=json.load(jsfile)
#ijson['data']['s']=-8
res=rm.predict(ijson['data'])
print(res['log'])

#path='D:\\ml\\'
#file='gpn_raw_v5.csv'
#dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
#columns=['ID простого участка','D', 'L', 'S','Дата ввода', 'Состояние','Дата перевода в бездействие', 'Дата аварии', 'Наработка до отказа',
       #'Адрес от начала участка', 'Обводненность','Дата окончания ремонта',
       #'Адрес от начала участка_1', 'Длина ремонтируемого участка','Дата ремонта до аварии',
       #'Адрес ремонта до аварии', 'Длина ремонта до аварии']
#rdata=pd.read_csv(path+file,parse_dates=dates,infer_datetime_format=True, dayfirst=True)
#ID=50007531
#group=rdata[rdata['ID простого участка']==ID]
#mdate=group['Дата аварии'].max()
#group.loc[:,'Дата аварии']=mdate
#group.loc[:,repcolumns]=np.nan
#pred=pml.predict(group,get='frame')
#print()

#np.save(path+'pred.npy',pred.values)