#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from os import listdir                                                 
from os.path import isfile, join
import textract

# Tokenize a file given it is a .txt file or a .pdf file
def tokenize_file(path):
    if '.txt' in path:
        fh = open(path,'r')
        file_text = fh.read()
    else:
        # For pdfs
        file_text=textract.process(path)
    return word_tokenize(file_text.decode('utf-8').lower())

# Tagg the tokens with a unique id
def tagg_tokens(tagged_tokens,data):
    start_index = 0
    # Try to get the last tagged_token
    if len(tagged_tokens)>0:
        start_index = int(tagged_tokens[-1].tags[0])+1

    # Tag the sentences with unique number ids --> maybe not the best!
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                                  tags=[str(i+start_index)]) for i, _d in enumerate(data)]
    # Extend the list
    tagged_tokens.extend(tagged_data)
    return tagged_tokens

# Folder for training files
folder = 'sources/'

#Training files
Training_files = [f for f in listdir(folder) if isfile(join(folder,f)) and (f.endswith('.txt') or f.endswith('.pdf'))]

# Tagg the data
tagged_data = []
for file_name in Training_files:
    print "Now reading: ", file_name
    tagged_data = tagg_tokens(tagged_data,tokenize_file(folder+file_name))
    
print "Done!"
print "Now training model..."
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1) # 1: distributed memory, 0:bag of words
  
model.build_vocab(tagged_data)

# Train
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

# Save the model
model.save("d2v.model")

print "Model saved!"

# -- Load and use model example --

# from gensim.models.doc2vec import Doc2Vec

# model= Doc2Vec.load("d2v.model")
# tweet = 'Undrar om majoriteten sov på mötet när deras egna tjänstemän använde just ordet vattenland som möjligt väg val istället för att bara renovera Tinnerbäcksbadet. Bra om de säger nej nu.'
# #to find the vector of a document which is not in training data
# test_data = word_tokenize(tweet.decode('utf-8').lower())
# v1 = model.infer_vector(test_data)
