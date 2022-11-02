import numpy as np

class rtd:
    def __init__(self):
        self.t=1.
        self.s=8.
        self.s0=4.
        self.sigma=0.
        self.mean=0.
        self.smin=4.
        self.measurements=np.array([])

    def fit(self,measurements=np.array([]),t=1.,s=8.,s0=4.):
        if (measurements is not None) and type(measurements) is np.ndarray:
            nneg=np.where(measurements<0)[0]
            assert nneg.shape[0]==0,'Некоторые значения измерений заданы отрицательными'
            assert measurements.size>0, 'Не заданы результаты толщинометрии'
            self.measurements=measurements
            self.sigma=self.measurements.std()
            self.smin=self.measurements.min()
            self.mean = self.measurements.mean()
        assert t>=0,"Задан отрицательный возраст трубопровода"
        assert s > 0, "Задана нулевая или отрицательная номинальная толщина стенки "
        assert s0 > 0, "Задана нулевая или отрицательная отбраковочная толщина стенки "
        self.t=t
        self.s=s
        self.s0=s0



    def value(self)->np.float32:
        tau=0
        if self.t==0:
            return 0
        size=self.measurements.size
        smin_=0
        if size<10:
            self.sigma=0
            smin_=self.smin
        else:
            smin=self.mean-2*self.sigma
            smin_=min(self.smin,smin)

        v=(self.s-smin_)/self.t
        tau=(smin_-self.s0)/v
        if tau<0:
            tau=0
        return tau

def predict(data,*args,**kwargs)->np.float32:
    measurements = np.array(data['measurements'], dtype=np.float32)
    s = np.array(data['s'], dtype=np.float32)
    s0 = np.array(data['s0'], dtype=np.float32)
    t = np.array(data['t'], dtype=np.float32)
    #undersampling = np.array(kargs['undersampling'], dtype=bool)

    model=rtd()
    model.fit(measurements=measurements,t=t,s=s,s0=s0)
    value=model.value()
    return value
