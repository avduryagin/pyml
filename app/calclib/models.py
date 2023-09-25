from generator import Generator
from generator import ClRe
import os
import pickle
import numpy as np

class SVRGenerator(Generator):
    def __init__(self,*args,classifier=None, regressor=None, col=None,path=None,modelfolder='models',regmodel='rfreg.sav',clmodel='rfc.sav',colfile='col.npy',scalers_folder='scalers',yscaler='yscaler.sav',scaler='scaler.sav',threshold=0.5,**kwargs):
        super().__init__(classifier=classifier, regressor=regressor, col=col,path=path,modelfolder=modelfolder,regmodel=regmodel,clmodel=clmodel,colfile=colfile)
        self.scaler_path=os.path.join(self.path,scalers_folder)
        self.yscaler=pickle.load(os.path.join(self.scaler_path,yscaler))
        self.scaler = pickle.load(os.path.join(self.scaler_path, scaler))
        self.threshold=threshold

    def get_next(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                              t=np.array([], dtype=float), s=np.array([], dtype=float), shape=np.array([], dtype=int)),
                 top=np.array([], dtype=float)):
        # прогнозирование класссификационной задачи
        prob = self.classifier.predict_proba(x.c)
        pred_mask = np.where(prob[:, 1] > 0.5)[0]
        # pred_mask = np.array(np.argmax(prob, axis=1), bool)
        # if pred_mask[pred_mask == True].shape[0] == 0:
        if pred_mask.shape[0] == 0:
            return None, pred_mask, prob
        # для  1 прогнозируется следующая точка y
        delta = self.regressor.predict(x.r[pred_mask]).reshape(-1)
        prev = x.r[pred_mask][:, -1]
        #delta = np.abs(y - prev)
        y = prev + delta
        emask = y == prev
        y[emask] = top[pred_mask][emask]
        y_hat = self.yscaler.inverse_transform(y.reshape(-1,1)).reshape(-1) * x.s[pred_mask]
        x_hat = x.get_items(mask=pred_mask)
        #r_tilde=x.r[:,0]
        x_hat.r[:, 0]=x_hat.r[:,0]+1
        x_hat.r[:,1]=y
        r_tilde=x_hat.r
        #r_tilde = np.hstack((x_hat.r[:, 1:], y.reshape(-1, 1)))
        x_tilde, t_tilde, shape_tilde = self.get_new(x=x_hat.c, tau=y_hat, t=x_hat.t, shape=x_hat.shape)
        return ClRe(c=x_tilde, r=r_tilde, t=t_tilde, shape=shape_tilde, s=x.s[pred_mask]), pred_mask, prob[:, 1]