from app import app
from app.calclib import pipeml
from flask_restful import Api, Resource, reqparse


api = Api(app)

class Calculation(Resource):
    def __init__(self, **kwargs):
        self.models=pipeml

    def post(self,*args,**kwargs):
        parser = reqparse.RequestParser()
        data = parser.parse_args()
        res=self.models.predict(data)
        return res

api.add_resource(Calculation, "/calc/")
