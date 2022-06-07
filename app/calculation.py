from app import app
from flask_restful import Api, Resource, reqparse

# Не используется пока
# from joblib import load

api = Api(app)

class Calculation(Resource):
    def __init__(self, **kwargs):
        self.models = kwargs['models']

    def find_models(self, well_id):
        found_models = []
        for model_name in self.models:
            for well in self.models[model_name]['wells']:
                if well == well_id:
                    found_models.append(model_name)
        return found_models

    # проверка на наличие возможности расчета
    def get(self, well_id):
        found_models = self.find_models(well_id)
        return len(found_models) > 0

    def post(self, well_id):
        parser = reqparse.RequestParser()
        parser.add_argument('data', required=True)

        data = parser.parse_args()['data']

        print(data)

        return {
            "id_simple_sector": 18175,
            "locate_simple_sector": 4480.0,
            "worl_avar_first": 14.2272035896,
        }

api.add_resource(Calculation, "/calc/<int:well_id>", resource_class_kwargs={'models': {
    'model2621cut': {
        'path': 'model2621truerateTEST_validated_20210326.h5',  
        'inputColumns': ['ACURR', 'LOADFACTOR', 'PUMPINTAKEPRESSURE', 'MOUTHPRESSURE', 'WORKINGFREQ', 'ACTIVEPOWER'],
        'wells': {
            8020262100: []
        }
    }
}})
