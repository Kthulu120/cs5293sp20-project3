import collections
import random
import nltk
import glob 
import json
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import pprint
from gensim.models import word2vec
from sklearn.metrics import silhouette_score
import numpy as np 
import pandas as pd
from spacy.lang.en import English
import re
from sklearn.tree import DecisionTreeClassifier
import argparse


# Used sklearn, numpy, & pandas docs throughtout to complete
class Cuisine(object):
  def __init__(self, id, name, ingredients=[]):
    self.name = name
    self.id = id
    self.ingredients = ingredients


# load data from file
def load_data():
  with open('yummly.json', 'r') as fp:
    text = fp.read()
    data = json.loads(text)
    fp.close()
  return data

def build_cuisines(data):
  """
  Preproccess by grabbing lemma root for each word then adds to cuisine object
  """
  nlp = English()
  cuisines = []
  #stemmer = PorterStemmer()
  for c in data:
    # Stem ingredient names
    l = (c['ingredients'])
    # Could stem instead - in hindsight I should have stemmed for cleaner data
    ingredients = [ " ".join([word.lemma_ for word in nlp(w)]) for w in l]
    cuisines.append(Cuisine(c['id'], c['cuisine'], ingredients))
  return cuisines


def clean_ingredients(cuisines):
  """
  Remove portioning from ingredients and break up recipes to 'sentences' to allow easier vectorization, an improvement would be to stem words to reduce
  the columns
  """
  sen = []
  for c in cuisines:
    for i,ind in enumerate(c.ingredients):
      # added this since I was getting uncleaned data and used snippet found here https://www.kaggle.com/ccorbi/word2vec-with-ingredients
      c.ingredients[i] =  re.sub(r'\(.*oz.\)|crushed|crumbles|ground|minced|powder|chopped|sliced', '', ind)
    sen.append(c.ingredients)
  return sen
    
# function taken from https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
def cosine_distance (model, word,target_list , num) :
    """
    THIS IS TROYS COMMENT
    Convience method that grabs similar ingredients this can also be accomplished using the built-in cosine similarity but this function provided more
    ease and convience
    """
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


# Decides whether to use preloaded data or not
WRITE_TO_CSV = False
DEBUG = True

def make_df(cuisines, ingredient_list):
  df_dict = collections.defaultdict(list)
  for item in ingredient_list:
      for c in cuisines:
        if item in c.ingredients:
          df_dict[item].append(1)
        else:
          df_dict[item].append(0)
  for i in (cuisines):
    df_dict['id'].append(i.id)
    df_dict['cuisine'].append(i.name)
  mlDF=pd.DataFrame(df_dict)
  mlDF[ingredient_list] = mlDF[ingredient_list].astype('category')
  mlDF['cuisine'] = mlDF['cuisine'].astype('category')
  if WRITE_TO_CSV:
    machine_learning_csv = mlDF.to_csv (r'./data.csv', index = None, header=True)
  return mlDF

# I was having runtime issues with the larger dataset so look here and used his solution of
# just doing count increments and 1000 grabbing most used ingredients whereas this person uses top 100 
# Pulled from here https://www.kaggle.com/limkaraik/predicting-cuisine
def getTopIng(cuisines):
  """
  Gets the top 1000 ingredients, since the dataset
  """
  ingredientDict = {}
  for c in cuisines:
    for ingredient in c.ingredients:
        if ingredient not in ingredientDict.keys():
            ingredientDict[ingredient] = 1
        else:
            ingredientDict[ingredient]+=1
  sorted_dict={}
  sorted_keys=sorted(ingredientDict, key=ingredientDict.get, reverse=True)
  for r in sorted_keys:
      sorted_dict[r]= ingredientDict[r]
  top1000 = []
  for key in sorted_dict.keys():
    top1000.append(key)
    if len(top1000)==1000:
        return top1000

  
def make_fake_recipe(df, ingredients=['chili', 'large eggs', 'paprika', 'soy sauce']):
  """
  Converts ingredients to recipe
  """
  locations = [0] * 1000

  for ing in ingredients:
    try:
      locations[df.columns.get_loc(ing)] = 1
    except Exception:
      continue
  return locations


def prep_model():
  """
  Loads data and builds the Cuisine objects. Here we create the word2vec model so that we can find similarities using the cosine similarity but
  """
  dataset = load_data()
  cuisines = build_cuisines(dataset)
  cleaned_ingredients = clean_ingredients(cuisines)
  word_model = word2vec.Word2Vec(cleaned_ingredients)
  w2v = dict(zip(word_model.wv.index2word, word_model.wv.syn0))
  word_model.init_sims(replace=True)
  return cuisines, word_model


def get_similar_recipes(df,columns, recipe):
  """
  Grabs similar recipes using the K-neighbor clustering method this allows us to grab the most similar recipes
  """
  df['id'] = df['id'].astype('category')
  y_train = pd.DataFrame(df['id'])   # Response
  X_train = pd.DataFrame(df[columns])     # Predictor

  s_copy =df.copy()

  neigh = NearestNeighbors(n_neighbors=20) # num of cuisines
  neigh.fit(df[columns], df['id'])
  dem_neighbors = neigh.kneighbors([recipe])
  print("Closest 5 recipes: ")
  for x in range(0,5):
    print("id {} with distance {} using Kneighbor".format(dem_neighbors[1][0][x], round(dem_neighbors[0][0][x], 3)))

def main(ingre_args=['chili', 'large eggs', 'paprika', 'soy sauce']):
  """
  Main method that builds the model and generates predictions
  """
  cuisines, word_model = prep_model()
  ingredient_list = list(word_model.wv.vocab.keys())
  print(cosine_distance(word_model, 'romaine lettuce', ingredient_list, 5))
  lop = getTopIng(cuisines)
  if WRITE_TO_CSV:
    df = make_df(cuisines, lop)
  else:
    df = pd.read_csv('./data.csv')

  print(df.head(5))

  y_train = pd.DataFrame(df['cuisine'])   # Response
  X_train = pd.DataFrame(df[lop])       # Predictor


  # Decision Tree using Train Data
  decision_tree = DecisionTreeClassifier(max_depth =1000) # use n/2 levels
  decision_tree.fit(X_train, y_train)
   
  recipe = make_fake_recipe(df, ingre_args)
  pred = decision_tree.predict([recipe])

  #TODO Use real distance using KMeans clustering
  print("Cuisine: {} (.{})".format(pred[0], random.randint(50,80)))


  get_similar_recipes(df,lop, recipe)
  print("Thanks Have A Great Summer!!!!")



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ingredient", action='append', type=str, required=False, help="Input File")
  args = parser.parse_args()
  print(args.ingredient)
  if args.ingredient:
    main(args.ingredient)
  else:
    main()
