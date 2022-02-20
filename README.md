# Brand Sentiment Model

Author: [Lia Elwonger](mailto:lelwonger@gmail.com)

## Overview

This project constructs an RNN to predict the sentiment of tweets towards Apple and Google products using data gathered from CrowdFlower found on [data.world](https://data.world/crowdflower/brands-and-product-emotions)

### Business Problem

Understanding the causes of negative or positive sentiment to newly released products can be helpful for both attempting to correct technical issue or aethetic features that hard sales. Twitter, which contains mountains of such potential information is however costly to sort through. Even searching simply for keywords associated with the brand are likely to find mountains of tweets with very little useful information and which might, for instance be discussing how they "googled" something rather than comments about the Google's band or products.

Advances in NLP can help with tis problem by allowing machines to do sorting of the data. It can be use to help in a number fo ways.

* Giving an estimate of the proportion of tweets under a hastag are saying something positive or negative about the brand.
* Giving a presort for market analysis by filtering out irrelevant tweets and saving human labor in market analysis.

### The Data

This project primarily uses data gathered from CrowdFlower who used crowdsourcing to identify the sentiment features of a set of tweets which can be found on found on [data.world](https://data.world/crowdflower/brands-and-product-emotions) or in the data folder in this project's GitHub repository. The data contains 9203 datapoints containing columns containing the full tweet, a column identifying what brand or product the tweet was about if any, and a final column indicating whether it has any emotion positive or negative or none towards the brand or product. The products found by keywrods seraching were all either Apple or Google products. The data was gathered in 2013 and all the tweets came from the #SXSW tag.

The relative smallness of the dataset from a NLP perspective will limit the overall achievable accuracy at this task, which is already a complex one. There is also significant class imbalance in the data with most of the data being marked as no emotion and less than 10% of the data expressing negative views towards the brand or product.

#### Preperation and Exploration
Rows that were marked "Could not tell" were changed to "No emotion" to simplify the analysis. 

Links, punctuation and stopwords were removed from the data prior to modeling. '#' was removed from twitter hashtags, but the content of the tag was kept. The data was small enough that lemmatization was usable to reduce the dimensionality of the data.

To get a general idea of the data visualizations for both the single word and bigram frequencies were made:

##### *Single Word Frequency*

![positive_word_cloud](/images/pos_frequency_cloud.png)

![negative_word_cloud](/images/neg_frequency_cloud.png)

![neutral_word_cloud](/images/neutral_frequency_cloud.png)

##### *Bigram Frequency*

![positive_bigram_word_cloud](/images/pos_bigram_frequency_cloud.png)

![negative__bigram_word_cloud](/images/neg_bigram_frequency_cloud.png)

![neutral_bigram_word_cloud](/images/neutral_bigram_frequency_cloud.png)

As we can see the differences in the data are very subtle.

##### *Mutual Information Score*

Difference in the mutual information scores of the bigrams in the different categories can also give a feel for how different the categories are:

![mutual_information_graph](/images/mutual_information.png)


### Modeling
The sentiment column was used as the target and the column marking which particular product was discussed was ignored since there is likely insufficient data to make good predictions for this columnd and using this in prediction of the sentiment would be infeasible in a real world application of the model. If you are going to have humans identify what product is discussed you might as well pay them to tell you the sentiment about it too, so this column could only reasonably be a target column.

With the relative smallness of the data it is potentially possible for fairly simple models to do about as well as a neural network on this dataset, so sklearn versions of these models were run on a TF-IDF vectorized version of the cleaned data with 1000 feature vectors.

#### *Bayesian Model*

![bayesian_matrix](/images/bayes_test_matrix.png)

#### *Logistic Model*

![logistic_matrix](/images/logistic_test_matrix.png)

#### *Random Forest Model*

![forest_matrix](/images/forest_test_matrix.png)

#### *SVC Model*

![SVC_matrix](/images/SVC_test_matrix.png)

#### *Baseline RNN*
A baseline neural network with an embedding layer, a LSTM layer, and a Dense layer with a dropout of 0.5 between layers was constructed using Keras to see if it would perform better on the data. See summary below for information on its construction:

summary

It did marginally better at about 65% accuracy, but perfomrmed very poorly on the rare negative category with a recall of only about 12% for that category, and despite the dropout between layers, began to signficantly overfit on traing data after only 3 epochs of training.

![baseline_matrix](/images/baseline_test_matrix.png)

To help resolve the class imbalance issue a weighted version of the model was run using weights gotten from the sklearn class weights utility. There was a marginal tradeoff in overall accuracy and F1 score for the weighted model over the non-weighed, but the weights significantly improved recall on the negative category.

![weighted_matrix](/images/weighted_test_matrix.png)

Additional improvements to the model were attempted including:

* Augmenting the data using replacement with synonyms
* Using a pretrained GloVe embedding rather than an untrained embedding layer.
* Increasing the size of the embedding
* Altering the number of nodes
* Altering the number of layers

### Results

Here are graphs of the performance of all the models according to accuracy, weighted F1 Score, and the F1 Score narrowly on the rare negative sentiment category:

![accuracy_graph](/images/accuracy_models.png)

![weighted_F1_graph](/images/f1_models.png)

![rare_F1_graph](/images/rare_f1_model.png)

There is no clear winner by all metrics, but the initial GloVe model perfroms best on accuracy and weighted f1, with only mild trade off on the negative category, and thus was saved as the final model

![glove_matrix](/images/glove_matrix.png)

#### *Model Interpretation*

-

### Limitations

Unfortunatly much of the benefits of neural networks comes with much larger dataset than were had here, and thiswas also a significantly difficult task for such a model given that unlike in review data there is no guarentee that approving or disapproving language is actually directed at the brnd or product, so the model needs to determine both that such sentiment exists and that it is directedto one of the possible brands or products that are inthe data, *and not others* which makes this task much more difficult.

### Conclusions

Doing large scale sentiment analysis on tweets is feasible, but only with sufficiently large and clear datasets. But, even with as littel data as is had here, a basic neural network can provide a useful filter for finding tweets or online mention that can be useful for market analysis.

## For More Information

Please review my full analysis in our Jupyter Notebook or our presentation.

For any additional questions, please contact **Lia Elwonger lelwonger@gmail.com**

## Repository Structure

```
├── README.md                           <- The top-level README for reviewers of this project
├── EDA_notebook.ipynb                  <- Notebook just containing the exploration and cleaning of the data
├── modeling_notebook.ipynb             <- Notebook containing just the modeling of the precleaned data
├── technical_notebook.ipynb            <- Narrative documentation of entire model construction process in Jupyter notebook
├── presentation.pdf                    <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code, includes cleaned data
├── code                                <- Functions for cleaning and processing the data and constructing visualizations
└── images                              <- Both sourced externally and generated from code
```
