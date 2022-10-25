from flask import request
from app import app
from app.calclib import pipeml
from app.calclib import remtime
import numpy as np
from flask_restful import Api, Resource
#import json


api = Api(app)

class Calculation(Resource):
    def __init__(self, **kwargs):
        self.models=pipeml

    def post(self,*args,**kwargs):
        x=request.json
        data = x['data']
        args = x['kwargs']
        try:
            model=x['model']
            if model=="rtime_df":
                self.models=remtime
                res=self.models.predict(data,args)

            res = self.models.predict(data, drift=args['drift'])
        except(KeyError):
            res = self.models.predict(data)


        return res

api.add_resource(Calculation, "/calc/")
