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

import flask_restplus
flask_restplus.__version__

import json
from Service import NERModel
from flask import Flask,request
from flask_restplus import Api,Resource,fields


import ssl 

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) 

#context.load_cert_chain('certificate.crt', 'private.key')

app = Flask(__name__)
api = Api(app=app,version='1.0', title='Valkyr-ie BIO',
          description='Using transformers')


name_space = api.namespace('bio-ner', description='BIO APIs')


    
Texto = api.model('Texto', {
    'text': fields.String(required=True, description='Text to be processed', default='Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia'),
})


NerModelNCBI = NERModel('NCBI','models/NCBI-disease/','models/NCBI-res/')
NerModelNCBI.initModel()


#NerModelJNLPBA = NERModel('JNLPBA','models/JNLPBA-disease/','models/JNLPBA-res/')
#NerModelJNLPBA.initModel()

#NerModelBC4CHEMD = NERModel('BC4CHEMD','models/BC4CHEMD-disease/','models/BC4CHEMD-res/')
#NerModelBC4CHEMD.initModel()

def generateBIOResponse(data,labels):
    
    #{'label':'O','word':'Hola'},{'label':'O','word':'Mundo'}
    List=[]
    
    for d,l in zip(data,labels):
        e ={ 'word':d,'label':l }
        List.append(e)
    return List
        
    
    



@name_space.route("/JNBPA/")
class JNBPA(Resource):
    
   
    
    @api.expect(Texto)
    def post(self):
        """
        To identify protein, DNA, RNA, cell line and cell type
        """
        data = request.json
        text = data.get('text')
        
        #TheSentence= ('Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia')
        
        #sentence, labels= NerModelJNLPBA.predict(text)
        
        response= {
         #   'tokens': generateBIOResponse(sentence,labels) 
            }
        
        return response
    
    
    
@name_space.route("/NCDIST/")
class NCIDIST(Resource):
    
    @api.expect(Texto)
    def post(self):
        """
        To identify Diseases
        """
        data = request.json
        text = data.get('text')
        
        #TheSentence= ('Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia')
        
        sentence, labels= NerModelNCBI.predict(text)
        
        response= {
            'tokens': generateBIOResponse(sentence,labels) 
            }
        
        return response
        
@name_space.route("/BC4CHEMD/")
class BC4CHEMD(Resource):
    
    @api.expect(Texto)
    def post(self):
        """
        To identify Chemicals and Drugs
        """
        data = request.json
        text = data.get('text')
        
        
        
        #sentence, labels= NerModelBC4CHEMD.predict(text)
        
        response= {
           # 'tokens': generateBIOResponse(sentence,labels) 
            }
        
        return response
    

if __name__ == '__main__':
    
    app.run(debug=True,ssl_context='adhoc', host='0.0.0.0',port=8088) 
    