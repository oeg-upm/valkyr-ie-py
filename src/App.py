# -*- coding: utf-8 -*-




"""
Created on Wed Mar 31 10:23:50 2021

@author: Pablo
"""






'''


!pip install flask

!pip install flask-restplus

!pip install Werkzeug==0.16.1


'''

import json
from Service import NERModel
from flask import Flask,request
from flask_restplus import Api,Resource,fields
app = Flask(__name__)
api = Api(app=app,version='1.0', title='My Blog API',
          description='A simple demonstration of a Flask RestPlus powered API')


name_space = api.namespace('bio-ner', description='BIO APIs')



    
Texto = api.model('Texto', {
    'text': fields.String(required=True, description='Category name'),
})


NerModel = NERModel('myNer','NCBI-disease/','mytest/')

NerModel.initModel()



@name_space.route("/JNBPA/")
class JNBPA(Resource):
    
   
   
    
    @api.expect(Texto)
    def post(self):
        print('entro')
        data = request.json
        print(data)
        text = data.get('text')
        print(text)
        """
        Adds a new conference to the list
        """
        TheSentence= ('Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia')
        
        
        
        sentence, labels= NerModel.predict(TheSentence)
        
        lis=[]
        for token,label in sentence,labels:
            w={ 'word':token, 
                
                }
        
        mys= {
          
            'words':[{'label':'O','word':'Hola'},{'label':'O','word':'Mundo'}]
            }
        
        return mys
    
    
    
@name_space.route("/NCDIST/")
class NCIDIST(Resource):
    def get(self):
        """
        returns a list of conferences
        """
    def post(self):
        """
        Adds a new conference to the list
        """

    

if __name__ == '__main__':
    app.run(debug=True,port=8088) 
    