# -*- coding: utf-8 -*-


"""
@author: Pablo Calleja
"""

import flask_restplus
import requests
from flask import Flask, request, current_app, abort,render_template,Blueprint, url_for
from flask_restplus import Api, Resource, fields,Namespace
from flask_swagger_ui import get_swaggerui_blueprint
#import request

'''
class MyApi(Api):
    @property
    def specs_url(self):
        """Monkey patch for HTTPS"""
        scheme = 'http' if '5000' in self.base_url else 'https'
        return url_for(self.endpoint('specs'), _external=True, _scheme=scheme)'
    '''


app = Flask(__name__)







blueprint = Blueprint("Valkyr-IE", __name__, url_prefix="/api")
api = Api(blueprint,version='1.0', title='Valkyr-IE', description='NLP Services')
app.register_blueprint(blueprint)






SWAGGER_UI_BLUEPRINT = get_swaggerui_blueprint(
    '/swagger',
    '/static/swagger.json',
    config={
        'app_name': "Valkyr-IE"
    }
)
app.register_blueprint(SWAGGER_UI_BLUEPRINT, url_prefix='/swagger')








#REQUEST_API = Blueprint("valkyr", __name__)






@app.route("/")
def index():
    return render_template("index.html")





#app.register_blueprint(REQUEST_API)
'''

@app.route("/templates")
def index():
    
    return render_template("index.html")

app.register_blueprint(apiblue)
'''
'''

Text = api.model('Text', {
    'text': fields.String(required=True, description='Text to be processed',
                          default='This software is was designed for Ubuntu and Pytorch 1.0'),
})
'''

'''
REQUEST_API = Blueprint("/",'/')
app.register_blueprint(REQUEST_API)
'''



ns_standard = api.namespace('ner-standard',description='State of the art Named Entity Recognition models')



Text = ns_standard.model('Text', {
    'text': fields.String(required=True, description='Text to be processed',
                          default='This software is was designed for Ubuntu and Pytorch 1.0'),
})



@ns_standard.route("/conll/")
class Model(Resource):

    @ns_standard.expect(Text)
    def post(self):
        """
        Method for detecting software libraries and requirements from text.
        Requirements include hardware requirements too. Anything on the "requirement" section
        """
        data = request.json
        text = data.get('text')
        print(text)
        return text



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8084)  # ssl_context='adhoc'
