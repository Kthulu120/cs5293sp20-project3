
### Setting Up
To setup install packages via `pip install -r requirements.txt`. 

### Example Command
`python ./csproject3/main.py --ingredient paprika --ingredient="soy sauce" --ingredient="sesame oil"`

### Running
Run `./csproject3/main.py` to run the main program, you run by repeating the `--ingredient` flag


#### How did you turn your text into features and why?
I turned each recipe into the object and then parsed those into a dataframe with the ingredient name as a value then when that ingredient is in a recipe we
assigned that cell the value 1 or 0 otherwise if the recipe doesn't contain that ingredient. We use the top 1000 ingredients since this takes up considerable processing time (39,744,000 iterations when unoptimized). Along with the ingredient the id and cuisine type are also columns. By constructing our features this way it allows us to load the dataset simply into our classifiers without doing any massive legwork beyond constructing the intial dataframe of recipes.

#### What classifiers/clustering methods did you choose and why?
I used the decision tree classifier, KNeighbors classifiers, along with word2Vec to find similarities between recipes. For prediction the Decision Tree with depth n (1000) allowed us to sufficiently guess the cuisine around 80% of the time therefore the decision seemed like a solid choice since we're working with finite likelihoods. Though this could be upped to 1000 for better predictions. I chose KNeighbors since I cared about the distance of the an ingredient between recipes and KNeighbors provided me the distance between the ingredients of recipes and a simple means of mapping back to the id for each neighbor group. 

#### What N did you choose and why?
I chose 5 recipes since it was in the assignment but also because in the KNeighbors contstructor we found the 20 closest neighbors so we could simply grab the 5 closest points to that recipe give some ingredients. 

#### Describe Function/Code
The program runs by launching the `main.py` file in the `csproject3` folder. It then prints the  
`make_fake_recipe` generates a fake row for the dataframe so we can easily run a prediction against the recipe given some ingredients.
`main` The main function takes in the input and grabs our dataset (or optionally load it from the json file if you don't want to use the csv I created). Then it generates a prediction using the decision tree classifier and prints the likely cuisine type. `make_df` loads the main dataframe after the dataset is parsed in the main method. `prep_model` creates the word2Vec model and returns it and the cuisine object list. The `get_similar_recipes` gets similar recipes using the KNearestNeighbor Algorithmn and prints them.

#### Tests
The way that I tested is importing my main file and running the program with some known ingredients and seeing if we get the expected cuisine type of Japansese given some different predetermined ingredients. Along with this we test similar recipes by checking the cuisine type of the the closest recipes. You can run the tests via `python ./tests.py`.
 NOTE: I had to literally import it due to import errors I kept getting.







##### Citations
[1] https://www.kaggle.com/ccorbi/word2vec-with-ingredients/data

[2] https://www.kaggle.com/limkaraik/predicting-cuisine

[3] https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92

[4] http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/