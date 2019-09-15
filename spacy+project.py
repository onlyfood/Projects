
# coding: utf-8

# In[1]:

import spacy


nlp=spacy.load('en_core_web_md')


# In[15]:

from spacy.matcher import Matcher

matcher1 = Matcher(nlp.vocab)
matcher2 = Matcher(nlp.vocab)

pattern1 = [{'ORTH': 'protein'},{'ORTH':'shake'}]

pattern2=[{'ORTH': 'gym'}]


matcher1.add("PROTEIN", None, pattern1)
matcher2.add("GYM", None, pattern2)


# In[26]:

TEXT1=['I woke up at 6am today.',{'id':'1'}]
TEXT2=['I live in vancouver.',{'id':'2'}]
TEXT3=['I go to gym by 6pm.',{'id':'3'}]
TEXT4=['After the gym work out I always enjoy a cup of protein shake',{'id':'4'}]

TEXT=[TEXT1,TEXT2,TEXT3,TEXT4]
TRAINING_DATA1 = []


for doc,context in nlp.pipe(TEXT,as_tuples=True):
    
    spans = [doc[start:end] for match_id, start, end in matcher2(doc)]
    
    entities = [(span.start_char, span.end_char, "GYM") for span in spans]
   
  
    training_example = (doc.text, {"entities": entities})
   
    TRAINING_DATA1.append(training_example)

print(*TRAINING_DATA1, sep="\n")


TRAINING_DATA2 = []

for doc,context in nlp.pipe(TEXT,as_tuples=True):
    
    spans = [doc[start:end] for match_id, start, end in matcher1(doc)]
    
   
   
    entities = [(span.start_char, span.end_char, "PROTEIN") for span in spans]
    
    training_example = (doc.text, {"entities": entities})
   
    TRAINING_DATA2.append(training_example)

print(*TRAINING_DATA2, sep="\n")


# In[24]:

nlp=spacy.blank('en')
ner=nlp.create_pipe('ner')
nlp.add_pipe(ner)
ner.add_label('PROTEIN')
ner.add_label('GYM')


# In[25]:



nlp.begin_training()


for itn in range(10):

   
    losses={}

   
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text,entities in batch]
        annotations = [entities for text, entities in batch]

      
      
        nlp.update(texts, annotations, losses=losses)
        print(losses)


# In[ ]:




# In[ ]:



