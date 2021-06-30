# -*- coding: utf-8 -*-


from flask import jsonify, abort, request, Blueprint
from flask import Flask,render_template
import requests
from flask_restplus import Resource, Api, fields, reqparse,Namespace
import json
from random import randint #libreria para random
import re
import os
from os import listdir
from os.path import isfile, isdir
#import time
#import termiup_terminal



REQUEST_API = Blueprint("valkyr", __name__)


# TO DO: read from properties/config yaml file.
VALKYRE_SERVICE = 'http://localhost:8084/processText'



def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route("/")
def index():
    return render_template("index.html")




ns = Namespace('ner', description='Cats related operations')


Text = ns.model('Text', {
    'text': fields.String(required=True, description='Text to be processed',
                          default='This software is was designed for Ubuntu and Pytorch 1.0'),
})

@ns.route("/requirements/")
class Requirements(Resource):

    @ns.expect(Text)
    def post(self):
        """
        Method for detecting software libraries and requirements from text.
        Requirements include hardware requirements too. Anything on the "requirement" section
        """
        data = request.json
        text = data.get('text')
        print(text)
        return text