import numpy as np
from scipy.stats import norm
import scipy.optimize as optim
import json
from functools import singledispatch

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


class pipe_parameters:
    def __init__(self):
        self._ryn=np.array(1,dtype=np.float32) #"R2 (Поле 'ГОСТ(ТУ), Завод изготовитель, Материал, Группа прочности') "
        self._run=np.array(1,dtype=np.float32) #"R1 (Поле 'ГОСТ(ТУ), Завод изготовитель, Материал, Группа прочности'
        self._d=np.array(0,dtype=np.float32)
        self._k=np.array(1,dtype=np.float32)
        self._p=np.array(0,dtype=np.float32) #Проектное давление МПа.
        self._pmax=np.array(0,dtype=np.float32)#Проектное давление МПа.
        self._c_h2s=np.array(0,dtype=np.float32) #Содержание сероводорода. По умолчанию равно 0.
        self._category=np.array(1,dtype=np.int32) #Категория работы трубопровода. Выбирается из множества {1,2,3}. По умолчанию 1 категория.
        self.conditions=dict({1: {"gamma_c":0.6,"pp_cond1":0.5,"pp_cond2":0.4},2:{"gamma_c":0.75,"pp_cond1":0.6,"pp_cond2":0.5},3:{"gamma_c":0.9,"pp_cond1":0.65,"pp_cond2":0.6}})
        self.permitted_cathegories=self.conditions.keys()
        self.isopened=True #Тип прокладки трубопровода. true  - при открытом типе укладки трубопровода. false - в противном.
        self._gas_percent=np.array(0,dtype=np.float32) #Газосодержание.
        self.gamma_n_array=np.array([[1,1,1,1,1.05],[1,1,1,1.05,1.1],
                                     [1,1,1.05,1.1,1.15],[1,1.05,1.1,1.15,0],
                                     [1.05,1.1,1.15,0,0],[1.1,1.15,0,0,0]],dtype=np.float32)
        self.pbounds=np.array([[0,7.5],[7.5,10],[10,15],[15,20],[20,np.inf]],dtype=np.float32)
        self.dbounds = np.array([[0, 400], [400, 600], [600, 800], [800, 1100], [1100,1300],[1300, np.inf]],dtype=np.float32)
        self._gamma_n=np.array(1,dtype=np.float32)
        self._gamma_s = np.array(0, dtype=np.float32)
        self._gamma_c = np.array(1, dtype=np.float32)
        self._gamma_m = np.array(1.55, dtype=np.float32)
        self._gamma_f = np.array(1, dtype=np.float32)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self,value:np.float32):
        assert value>=0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент несущей способности трубопровода {0:.2f}".format(value)+ ". Допускется только неотрицательное значение."
        self._k = value

    @k.deleter
    def k(self):
        del self._k

    @property
    def gamma_s(self):
        return self._gamma_s

    @gamma_s.setter
    def gamma_s(self,value:np.float32):
        assert value>=0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент надежности трубопровода {0:.2f}".format(value)+ ". Допускется только неотрицательное значение."
        self._gamma_s = value

    @gamma_s.deleter
    def gamma_s(self):
        del self._gamma_s

    @property
    def gamma_n(self):
        return self._gamma_n

    @gamma_n.setter
    def gamma_n(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент надежности трубопровода {0:.2f}".format(value)+ ". Допускется только положительное значение."
        self._gamma_n = value

    @gamma_n.deleter
    def gamma_n(self):
        del self._gamma_n

    @property
    def gamma_m(self):
        return self._gamma_m

    @gamma_m.setter
    def gamma_m(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент надежности трубопровода по материалам {0:.2f}".format(value)+ ". Допускется только положительное значение."
        self._gamma_m = value

    @gamma_m.deleter
    def gamma_m(self):
        del self._gamma_m

    @property
    def gamma_f(self):
        return self._gamma_f

    @gamma_f.setter
    def gamma_f(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент надежности трубопровода по нагрузке {0:.2f}".format(value)+ ". Допускется только положительное значение."
        self._gamma_f = value

    @gamma_f.deleter
    def gamma_f(self):
        del self._gamma_f

    @property
    def gamma_c(self):
        return self._gamma_c

    @gamma_c.setter
    def gamma_c(self,value:np.float32):
        assert value>=0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Коэффициент условий работы трубопровода {0:.2f}".format(value)+ ". Допускется только неотрицательное значение."
        self._gamma_c = value

    @gamma_c.deleter
    def gamma_c(self):
        del self._gamma_c

    @property
    def ryn(self):
        return self._ryn

    @ryn.setter
    def ryn(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Задан некорректный параметр R2 (Ryn). Допускается положительное действительное число"
        self._ryn = value

    @ryn.deleter
    def ryn(self):
        del self._ryn


    @property
    def run(self):
        return self._run

    @run.setter
    def run(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Задан некорректный параметр R1(Run). Допускается положительное действительное число"
        self._run = value

    @run.deleter
    def run(self):
        del self._run



    @property
    def d(self):
        return self._d

    @d.setter
    def d(self,value:np.float32):
        assert value>0 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Задан некорректный внешний диаметр. Допускается положительное действительное число"
        self._d = value

    @d.deleter
    def d(self):
        del self._d

    @property
    def c_h2s(self):
        return self._c_h2s

    @c_h2s.setter
    def c_h2s(self,value:np.float32):
        assert value>=0 and value<=1 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Доля сероводорода задана некорректно "
        self._c_h2s = value

    @c_h2s.deleter
    def c_h2s(self):
        del self.c_h2s

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self,value:np.float32):
        assert value>0  and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Задано отрицательное давление "
        self._p = value

    @p.deleter
    def p(self):
        del self._p


    @property
    def pmax(self):
        return self._pmax

    @pmax.setter
    def pmax(self,value:np.float32):
        assert value>0  and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Задано отрицательное давление проектное давление"
        self._pmax = value

    @pmax.deleter
    def pmax(self):
        del self._pmax

    @property
    def gas_percent(self):
        return self._gas_percent

    @gas_percent.setter
    def gas_percent(self,value:np.float32):
        assert value>=0 and value<=1 and isinstance(value,(np.float32,float,np.float64,np.ndarray)),"Газовый фактор задан некорректно"
        self._gas_percent = value

    @gas_percent.deleter
    def gas_percent(self):
        del self._gas_percent

    @property
    def cathegory(self):
        return self._category
    @cathegory.setter
    def cathegory(self,value:np.int32):
        cat=np.isin(value,list(self.permitted_cathegories))
        assert cat==True, "Задана категория вне допустимого диапазона"
        self._category=value
    @cathegory.deleter
    def cathegory(self):
        del self._category

    @property
    def opened_kind(self):
        return self.isopened
    @opened_kind.setter
    def opened_kind(self,value:bool):
        assert isinstance(value,(bool,np.ndarray)), " Задан недопустимый тип прокладки трубопровода"
        self.isopened=value
    @opened_kind.deleter
    def opened_kind(self):
        del self.isopened

    def get_gamma_n(self):

        def get_pi()->np.int32:
            i=np.int32(0)
            while i<self.pbounds.shape[0]:
                a=self.pbounds[i,0]
                b = self.pbounds[i, 1]
                if (self.p>=a)&(self.p<b):
                    return i
                i+=1
            return np.nan

        def get_di()->np.int32:
            i=np.int32(0)
            while i<self.dbounds.shape[0]:
                a=self.dbounds[i,0]
                b = self.dbounds[i, 1]
                if (self.d>=a)&(self.d<b):
                    return i
                i+=1
            return np.nan
        pindex=get_pi()
        dindex=get_di()
        if (~np.isnan(pindex))&(~np.isnan(dindex)):
            self.gamma_n=self.gamma_n_array[dindex,pindex]
        else:
            assert ~np.isnan(pindex),'Рабочее давление {0:.2f}'.format(self.p)+" выходит за допустимые границы расчета"
            assert ~np.isnan(dindex), 'Внешний диаметр трубопровода {0:.2f}'.format(self.d) + " выходит за допустимые границы расчета"






    def get_gamma_s(self):
        coef=np.float32(0.)
        cat=self.cathegory
        if self.c_h2s>0:
            pp=self.pmax*self.c_h2s/100.

            if (pp>=3e-4)&(pp<0.1):
                coef=self.conditions[cat]['pp_cond1']
            elif (pp>=0.1)&(pp<=1):
                coef = self.conditions[cat]['pp_cond2']
            else:
                assert pp<=1,"Значение парциального давления "+'{0:.3f}'.format(pp)+" выходит за рамки расчета."
        self.gamma_s=coef




    def get_gamma_c(self):
        cat=self.cathegory
        self.gamma_c= self.conditions[cat]['gamma_c']


    def get_gamma_f(self):
        coef=1
        a=1
        b=1
        if self.gas_percent==1:
            a=1.1
            b=1.1
        elif self.gas_percent==0:
            a=1.15
            b=1.
        else:
            b=1.1
            a=b*1.15

        #coef=1.1*1.2*b*a*1.1*1.5
        coef=2.178*a*b

        self.gamma_f=np.float32(coef)

    def fit(self,*kargs,ryn=1,run=1,p=0,pmax=0,kind=True,cathegory=1,d=100,c_h2s=0,gas_percent=0,k=1,gamma_m=1.55,**kwargs):
        self.opened_kind=kind
        self.ryn=ryn
        self.run=run
        self.p=p
        self.pmax=pmax
        self.cathegory=cathegory
        self.d=d
        self.c_h2s=c_h2s
        self.gas_percent=gas_percent
        self.k=k
        self.get_gamma_c()
        self.get_gamma_n()
        self.get_gamma_s()
        self.get_gamma_f()
        self.gamma_m=gamma_m

    def value(self)->np.float32:
        if self.gamma_s>0:
            r=self.run*self.gamma_n/self.gamma_n
        else:
            r=min(self.run*self.gamma_c/(self.gamma_m*self.gamma_n),self.ryn*self.gamma_c/(0.9*self.gamma_n))

        x=2*(r+0.6*self.gamma_f*self.p)
        sr=self.gamma_f*self.k*self.p*self.d/x
        return sr







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

    def predict(self,s_=8.,ds=0.04,m=0.05,q=0.99,z=100):

        assert (self.measurements.size >= 4), "Задано недотаточное для выполнения расчетов количество измерений"
        assert s_>=0,"Задана нулевая или отрицательная расчетная толщина стенки"
        assert ds > 0, "Задано отрицательое относительное начальное (технологическое) среднеквадратическое отклонение стенки"
        assert m >= 0, "Задана отрицательное число допустимых отказавших элементов в год на 1 км длины "
        assert z >= 0, "Задана отрицательное число элементов трубопровода на 1 км"
        assert (q >= 0) and (q<1), "Задана некоректная доверительная вероятность. Значение должно быть положительным меньше 1. "


        n=self.measurements.size
        self.z=z
        self.s_=s_
        self.q=q
        self.m=m
        self.ds2=ds**2
        self.dmean=1-self.mean/self.s
        self.su=self.sigma/self.s
        self.quantile=norm.ppf(q)
        self.maxsq2=(self.su*(1+self.quantile/(np.power(2*n-8,0.5))))**2
        self.delta=self.dmean+self.quantile*(self.su/np.power(n-2,0.5))
        self.sk2=(self.maxsq2-self.ds2)/(self.t**2)
        self.dr=1-self.s_/self.s
        self.r=self.delta/self.t
        self.dtau=self.dr-self.delta
        self.epsilon=self.m/self.z
        self.args=[]


        def value(i=0)->np.float32:
            x1=self.dtau-self.r*(i)
            x2=x1-self.r
            ksi=self.t+i
            i2=(ksi)**2
            i3=i2+2*ksi+1
            y1=np.power(self.sk2*i2+self.ds2,0.5)
            y2=np.power(self.sk2*i3+self.ds2,0.5)
            w1=x1/y1
            w2=x2/y2
            self.args.append([w1,w2])
            #print(w1)
            #print(w2)
            return np.abs(self.epsilon-(norm.cdf(w1)-norm.cdf(w2)))

        return value

    def get_time(self,s_=8.,ds=0.04,m=0.05,q=0.99,z=100,tol=1e-2):
        fun=self.predict(s_=s_,ds=ds,m=m,q=q,z=z)
        val=optim.minimize_scalar(fun,method='bounded',bounds=(0,30),tol=tol)
        return val





class techstate:
    def __init__(self):
        self.s=8.
        self.s0=4.
        self.tfirst=0.
        self.time=np.array([],dtype=np.float32)
        self.width=np.array([],dtype=np.float32)
        self.values = np.array([], dtype=np.float32)
        self.sa=np.array([],dtype=np.int16)
        self.vcorr=0.
        self.vcorr_mean=0.
        self.vcorr_fact=0.
        self.vcorr_max=0.

    def concatinate(self,l):
        def get_length(l)->np.int32:
            if l is None:
                return 0
            if (type(l) == list) or (type(l) == tuple):
                n = len(l)
            elif type(l) == np.ndarray:
                n = l.shape[0]
            else: n=0
            return n


        n=get_length(l)
        length=np.array([get_length(x) for x in l],dtype=np.int32)
        m = np.max(length)
        if m==0:
            if type(l)==np.ndarray:
                return l.reshape(-1,1)
            else:
                return np.array(l,dtype=np.float32).reshape(-1,1)



        array=np.empty(shape=(n,m),dtype=np.float32)
        array.fill(np.nan)
        i=0
        while i<n:
            l_=l[i]
            size=length[i]
            if 0==size:
                array[i,0]=l_
                i+=1
                continue
            for j,v in enumerate(l_):
                array[i,j]=v
            i+=1
        return array



    def fit(self, s=np.array(8.), time=np.array([]), width=np.array([]), tfirst=np.array(0.), s0=np.array(4.)):

        if (time is not None) and type(time) is np.ndarray and (width is not None) and type(width) is np.ndarray:
            wneg=np.where(width<0)[0]
            tneg = np.where(time < 0)[0]
            assert wneg.shape[0]==0,'Некоторые значения измерений заданы отрицательными'
            assert tneg.shape[0] == 0, 'Некоторые значения времени заданы отрицательными'
        else:
            assert False, "Задан некорректный тип данных. Ожидается numpy.ndarray"
        assert tfirst>=0,"Задан отрицательный возраст первой аварии"
        assert s > 0, "Задана нулевая или отрицательная номинальная толщина стенки "
        assert s0 > 0, "Задана нулевая или отрицательная отбраковочная толщина стенки "



        if (time.shape[0]==width.shape[0]) and time.shape[0]>0:
            self.time=time
            self.width=width
        elif (time.shape[0]>0) and (width.shape[0]>0) and (time.shape[0]!=width.shape[0]):
            n=min(time.shape[0],width.shape[0])
            self.time=time[:n]
            self.width=width[:n]
        if self.time.shape[0]>0:
            self.sa=np.argsort(self.time)
            self.time=self.time[self.sa]
            self.width=self.width[self.sa]
        self.tfirst=tfirst
        self.s=s
        self.s0=s0

    def get_samples(self,i: np.int16) -> np.ndarray:
        array = np.zeros(4)
        if i >= self.time.shape[0]:
            return array
        index = -(i + 1)
        cwidth = np.nanmin(self.width[index])
        ctime = self.time[index]
        array[0] = ctime
        array[1] = cwidth
        try:
            pwidth = np.nanmin(self.width[index - 1])
            ptime = self.time[index - 1]
            array[2] = ptime
            array[3] = pwidth

        except IndexError:
            array[3]=self.s
            return array
        finally:
            return array

    def get_values(self):
        def get_vcorr(cwidth=1,ctime=1,pwidth=0,ptime=0):
            dt = ctime - ptime
            if dt > 0:
                vcorr = (pwidth - cwidth) / dt
                if vcorr > 0:
                    return vcorr
            return 0

        def get_vmax():
            values=[]
            i=0
            maxval=0
            index=0
            while i<self.time.shape[0]:
                smaple = self.get_samples(i)
                ctime = smaple[0]
                cwidth = smaple[1]
                ptime = smaple[2]
                pwidth = smaple[3]
                val=get_vcorr(cwidth=cwidth, ctime=ctime, pwidth=pwidth, ptime=ptime)
                values.append(val)
                if maxval<val:
                    maxval=val
                    index=i
                i+=1
            return index,np.array(values,dtype=np.float32)


        if self.time.shape[0]==0:
            return self.vcorr,self.vcorr_mean,self.vcorr_fact,self.vcorr_max
        latest=self.get_samples(0)
        ctime = latest[0]
        cwidth=latest[1]
        #ptime = latest[2]
        #pwidth=latest[3]

        index,values=get_vmax()
        self.values=values
        self.vcorr=values[0]
        self.vcorr_max=values[index]


        if ctime>0:
            vcorr_mean=(self.s-cwidth)/ctime
            if vcorr_mean>0:
                self.vcorr_mean=vcorr_mean

        if self.tfirst>0:
            vcorr_fact=(cwidth-self.s0)/self.tfirst
            if vcorr_fact>0:
                self.vcorr_fact=vcorr_fact

        return self.vcorr,self.vcorr_mean,self.vcorr_fact,self.vcorr_max




def predict(data,*args,**kwargs)->np.float32:
    measurements = np.array(data['measurements'], dtype=np.float32) #последние измерения дефектоскопии на трубопроводе.
    #s = np.array(data['s'], dtype=np.float32) #номинальная толщина.
    s = np.float32(data['s'])  # номинальная толщина.
    #sestimated = np.array(data['sestimated'], dtype=np.float32) #расчетная толщина.
    tol = np.float32(data['tolerance'])#точность расчетов
    m = np.float32(data['acceptable'])  #допустимое число отказов в год на 1 км длины
    z = np.float32(data['nelements'])  # число элементов трубопровода на на 1 км длины (по умолчанию 100)
    q = np.float32(data['proba'])  # доверительная вероятность
    #- 0,99 – для напорных нефтепроводов, газопроводов и водоводов высокого давления,
    #- 0,95 – для нефтесборных коллекторов, трубопроводов выкидных линий и водоводов низкого давления.

    ds = np.float32(data['sigma0'])  # относительное среднее квадратичное начальное
    run = np.float32(data['run'])  # минимальное значение временного сопротивления, Н/мм^2.
    ryn = np.float32(data['ryn'])  # минимальное значение предела текучести материала, Н/мм^2.
    # отклонение толщины стенки трубы в начальный момент,
    # равное 0,04 для бесшовных труб и 0,03 – для сварных труб;
    s0 = np.float32(data['s0']) #отбраковочная толщина.
    d = np.float32(data['d'])  # внешний диаметр трубы, мм.
    p = np.float32(data['p'])  # рабочее давление,  МПа.
    pmax= np.float32(data['pmax'])  #максимальное давление,  МПа.
    ch2s = np.float32(data['ch2s'])  #содержание в газе сероводорода в объемных процентах.
    gas_percent = np.float32(data['gas_percentage'])  #Газосодержание, доля.
    k = np.float32(data['k'])  # Газосодержание, доля.
    opened = bool(data['opened'])  # Тип прокладки трубопровода.
    cathegory = np.int32(data['cathegory']) #Категория трубопровода.
    t = np.float32(data['t']) #возраст трубопровода на момент последней диагностики.
    tfirst=np.float32(data['tfirst']) #возраст перой аварии по коррозии.
    time=np.array(data['samples']['time'], dtype=np.float32) # массив, содержащий возраст трубопровода на момент проведения диагностик.
    width = np.array(data['samples']['width'], dtype=np.float32) #измерения дефектоскопии в контрольной точке,
    # соотнесенные с массивом time.

    #undersampling = np.array(kargs['undersampling'], dtype=bool)

    results=dict({"probab_rtime":0,"vcorr_by_meas":0,"vcorr_mean":0,"vcorr_mean_fact":0,"vcorr_max":0,"predicted_rtime":0,"log":""})
    try:
        properties=pipe_parameters()
        properties.fit(run=run, ryn=ryn, p=p, pmax=pmax, cathegory=cathegory, kind=opened, d=d, c_h2s=ch2s,
                   gas_percent=gas_percent, k=k)
        sestimated = properties.value()
        model=rtd()
        model.fit(measurements=measurements,t=t,s=s,s0=s0)
        condtau=model.value()
        results["probab_rtime"]=float(condtau)
        val = model.get_time(s_=sestimated, tol=tol,q=q,z=z,m=m,ds=ds)
        tau=val['x']
        results["predicted_rtime"]=float(tau)
        tech = techstate()
        width = tech.concatinate(width)
        tech.fit(width=width, time=time, s0=s0, s=s, tfirst=tfirst)
        vcorr,vcorr_mean,vcorr_fact,vcorr_max=tech.get_values()
        results["vcorr_by_meas"]=float(vcorr)
        results["vcorr_mean"] = float(vcorr_mean)
        results["vcorr_mean_fact"] = float(vcorr_fact)
        results["vcorr_max"] = float(vcorr_max)

    except AssertionError as error:
        results['log']=error.args[0]


    tojs=results
    #tojs = json.dumps(results, default=to_serializable)


    return tojs


