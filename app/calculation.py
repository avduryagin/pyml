from flask import request
from app import app
from app.calclib import pipeml
from flask_restful import Api, Resource
import json


api = Api(app)

class Calculation(Resource):
    def __init__(self, **kwargs):
        self.models=pipeml

    def post(self,*args,**kwargs):
        x = json.dumps(request.json)
        res=self.models.predict(x)
        return res

api.add_resource(Calculation, "/calc/")
