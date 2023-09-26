import pandas as pd
import numpy as np
import app.calclib.pipeml as pml
import app.calclib.remtime as remtime
import app.calculation as calc
import os
import remtime as rm
import json

import app.calclib.repair_programms as rp
path="C:\\Users\\avduryagin\\etc"
file='inscribed.csv'
dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
xdata=pd.read_csv(os.path.join(path,file),parse_dates=dates,infer_datetime_format=True, dayfirst=True,engine='c')
group=xdata[xdata['ID простого участка']==62468]
mask=group['Дата аварии']>=np.datetime64('2015-01-01')

cov=rp.pipe_cover(group,mask=mask)
cov.fit()