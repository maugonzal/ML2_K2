#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import keras
from m09_model_deployment import predict_proba
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app,
    version='1.0',
    title='API Predicción de Precio',
    description='API para la predicción del vehículo usado'
)

ns = api.namespace('predict', description='Estimación Precio')

parser = api.parser()

parser.add_argument(
    'X',
    type=str,
    required=True,
    help='Variables del vehículo',
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        return {
            "result": predict_proba(args['X'])
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)