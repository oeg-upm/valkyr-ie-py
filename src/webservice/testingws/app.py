# -*- coding: utf-8 -*-


import flask_restplus
import requests
from flask import Flask, request, current_app, abort,render_template,Blueprint
from flask_restplus import Api, Resource, fields
from flask_swagger_ui import get_swaggerui_blueprint



app = Flask(__name__)
#api = Api(app=app, version='1.0', title='Valkyr-IE', description='Named Entity Recognition',prefix='/docs')
#name_space = api.namespace

@app.route("/")
def index():
    
    return render_template("index.html")







if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # ssl_context='adhoc'
