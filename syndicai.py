
from flair.models.text_classification_model import TextClassifier
from flair.data import Sentence
import pandas as pd
import numpy as np
import html
import random
from IPython.core.display import display, HTML
from spacy.lang.en import English
import anvil.server
import wget
anvil.server.connect("I7N2M2OP46BP74QKPAPNDCGP-T2FVRS5JEOANDXEG")

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
nlp.add_pipe('sentencizer')
result=[]

overall_score= 0

class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
#        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()

        #self.classifierTyp = TextClassifier.load('/content/drive/MyDrive/type_disease_trans_more/final-model.pt')
        #self.classifierMis = TextClassifier.load('/content/drive/MyDrive/pure_corpus_17k_slow_bert_latest/final-model.pt')
        
        
        
        wget.download(
            "https://drive.google.com/drive/folders/1lKTy9hqEn3xXh-tYQZlUx8m2rRKtiCcv?usp=sharing",
            "/tmp/classifierMis/final-model.pt",
        )
        wget.download(
            "https://drive.google.com/drive/folders/10--DeptxxFgEKWBL877yteytyf6CE_VS?usp=sharing",
            "/tmp/classifierTyp/final-model.pt",
        )
        self.classifierTyp = TextClassifier.load("/tmp/classifierTyp/final-model.pt")
        self.classifierMis = TextClassifier.load("/tmp/classifierMis/final-model.pt")
        
        
    # Prevent special characters like & and < to cause the browser to display something other than what you intended.
    def html_escape(self, payload):
        return html.escape(payload)

    def sentence_fun(self, payload):
        #print('Hisentencesssss')
        text = payload
    #  "nlp" Object is used to create documents with linguistic annotations.
        doc = nlp(text)
    # create list of sentence tokens
        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text)
            result.append(sent.text)
            #print(sent.text)
        return sents_list

    # Prevent special characters like & and < to cause the browser to display something other than what you intended.
    def html_escape(self, payload):
        return html.escape(payload)
    
    @anvil.server.callable
    def predict(self, payload):
        #print('Hisentence')
        result=[]
        sents_list=sentence_fun(payload) # sentence_fun(text)
        myresult = pd.DataFrame({'sentence': sents_list}) #myresult = pd.DataFrame({'sentence': result})
        r = myresult['sentence'].replace('\n','', regex=True)
        r = r.replace('\n',' ', regex=True)
        r = r.replace(r'\\n',' ', regex=True)
        myresult['sentence']=r
        myresult
        ###############################
        sample_sent=[]
        for i in range(len(myresult['sentence'])):
            sentence = Sentence(myresult['sentence'][i])
            sample_sent.append(sentence)
            #print(sentence)
        sample_sent
        ###############################

        sample_annotated=[]
        for i in range(len(myresult['sentence'])):
            self.classifierTyp.predict(sample_sent[i])
            sample_annotated.append(sample_sent[i].labels)

        sample_annotated

        scory=[]
        labelnew=[]
        for i in range(len(myresult['sentence'])):
            scory.append(sample_sent[i].labels[0].to_dict()['confidence'])
            labelnew.append(sample_sent[i].labels[0].to_dict()['value'])

        myresult['label_type']=labelnew
        myresult['score_type']=scory

        myresult

        #############################
        sample_annotated2=[]
        for i in range(len(myresult['sentence'])):
            self.classifierMis.predict(sample_sent[i])
            sample_annotated2.append(sample_sent[i].labels)

        sample_annotated2

        scory2=[]
        labelnew2=[]
        for i in range(len(myresult['sentence'])):
            scory2.append(sample_sent[i].labels[0].to_dict()['confidence'])
            labelnew2.append(sample_sent[i].labels[0].to_dict()['value'])

        myresult['label_mis']=labelnew2
        myresult['score_mis']=scory2
        pd.set_option('display.max_colwidth', None)
        myresult

        ###########################################################################
        avgscory=[]
        for i in range(len(myresult['sentence'])):
            if (myresult['label_mis'][i] =='FAKE'):
                myresult['score_mis'][i] = -myresult['score_mis'][i]
            avg_score=(myresult['score_mis'][i])
            print('Hi im avg_score',avg_score)
            avgscory.append(avg_score )
            
        myresult['avg_score']=avgscory
        myresult

        ###############################################
        df_coeff = myresult[['sentence','avg_score','label_type','score_type']]
        df_coeff
        ########################################
        if (len(df_coeff['avg_score']) == 1):
            df_coeff['normalized_score']=(df_coeff['avg_score']+1 )/2
            print('normalized 1 record',df_coeff['normalized_score'][0])
        else:
            #df_coeff['normalized_score']=(df_coeff['avg_score']-df_coeff['avg_score'].min())/(df_coeff['avg_score'].max()-df_coeff['avg_score'].min())
            df_coeff['normalized_score']=(df_coeff['avg_score']+1)/(2)
            print('normalized Multi records 0 =',df_coeff['normalized_score'][0])
            print('normalized Multi records 1 =',df_coeff['normalized_score'][1])
        ############################################
        df_coeff["label_type"] = "[" + df_coeff["label_type"] + "]."
        df_coeff['normalized_full']=(df_coeff['avg_score']+1)/(2) # normalized on full range from -1 to +1
        overall_score=df_coeff["normalized_full"].mean()
        overall_score=round(overall_score,4)
        print(df_coeff)
        
        ############################################
        # https://adataanalyst.com/machine-learning/highlight-text-using-weights/
        max_alpha = 0.8 
        highlighted_text = []
        for i in range(len(df_coeff)):
            weight = df_coeff['normalized_score'][i]
            weight_type = df_coeff['score_type'][i]
            
            if weight is not None:
                highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(weight / max_alpha) + ');">' + html_escape(df_coeff['sentence'][i]) + '</span>')
                highlighted_text.append('<span style="background-color:rgba(255,255,0,' + str(weight_type / max_alpha) + ');">' + html_escape(df_coeff['label_type'][i]) + '</span>')
            else:
                highlighted_text.append(df_coeff['sentence'][i])
                highlighted_text.append(df_coeff['label_type'][i])
        highlighted_text = ' '.join(highlighted_text)
        #############################################################################
        #print(highlighted_text)
        return highlighted_text,overall_score

    @anvil.server.callable
    def predict_iris2(text):
        return text

#highlighted_text = []

#highlighted_text,overall_score=predict('Drink water to prevent heart disease. Drink water to not prevent heart disease.')
#highlighted_text
#overall_score

#display(HTML(highlighted_text))
