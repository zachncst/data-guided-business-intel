import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import numpy as np
from sets import Set
random.seed(0)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg
 

def build_word_map_count( in_list ) :
    out_map = {}

    for text in in_list:
        used_word = Set()
        for word in text:
            if word in out_map: #and word not in used_word:
                out_map[word] += 1
            else:
                out_map[word] = 1

            #used_word.add(word)

    return out_map

def filter_map_keys(in_map, filter_coll):
    new_map= {}
    filter_coll_set = Set(filter_coll)

    for word in in_map:
        if word not in filter_coll_set:
            new_map[word] = in_map[word]

    return new_map

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    #Collect a map of words to the number of texts they were featured in, a word used twice
    #in the same text is only counted once
    pos_word_map = build_word_map_count( train_pos + test_pos )
    neg_word_map = build_word_map_count( train_neg + test_neg )

    #Count total texts
    total_pos = len( train_pos + test_pos )
    total_neg = len( train_neg + test_neg )

    #Filter out any words in stopwords
    pos_word_map_filtered = filter_map_keys(pos_word_map, stopwords)
    neg_word_map_filtered = filter_map_keys(neg_word_map, stopwords)

    #Collect all positive words with >= 1% and is in twice as many positive texts than negative
    #print pos_word_map_filtered
    feature_list_pos = [k for (k,v) in pos_word_map_filtered.iteritems()
                    if v >= 0.01*total_pos
                    and v >= 2 * neg_word_map_filtered[k]
    ]
    feature_list_neg = [k for (k,v) in neg_word_map_filtered.iteritems()
                    if v >= 0.01*total_neg
                    and v >= 2 * pos_word_map_filtered[k]
    ]
    feature_list = Set(feature_list_pos + feature_list_neg)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.

    def has_feature(words):
        word_set = Set(words)
        return reduce(lambda x,y: x+y, map(lambda feat:[1 if feat in word_set else 0], feature_list))

    train_pos_vec = map(lambda w: has_feature(w), train_pos)
    train_neg_vec = map(lambda w: has_feature(w), train_neg)
    test_pos_vec  = map(lambda w: has_feature(w), test_pos)
    test_neg_vec  = map(lambda w: has_feature(w), test_neg)

    # Return the four feature vectors

    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def labeled_sentence_builder(wordz, label):
    return LabeledSentence(words=wordz['sentence'], tags=[label])

def build_feature_vector(label_format, length, model):
    my_result = []

    for i in range(length):
        my_label = label_format.format(i)
        my_result.append(model.docvecs[my_label])

    return my_result

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    index_train_pos  = [{'sentence' : words, 'rank' : i} for i,words in enumerate(train_pos)]
    index_train_neg  = [{'sentence' : words, 'rank' : i} for i,words in enumerate(train_neg)]
    index_test_pos   = [{'sentence' : words, 'rank' : i} for i,words in enumerate(test_pos)]
    index_test_neg   = [{'sentence' : words, 'rank' : i} for i,words in enumerate(test_neg)]

    labeled_train_pos = map(lambda words: labeled_sentence_builder(words, "train_pos_{}".format(words['rank'])), index_train_pos)
    labeled_train_neg = map(lambda words: labeled_sentence_builder(words, "train_neg_{}".format(words['rank'])), index_train_neg)
    labeled_test_pos  = map(lambda words: labeled_sentence_builder(words, "test_pos_{}".format(words['rank'])),  index_test_pos)
    labeled_test_neg  = map(lambda words: labeled_sentence_builder(words, "test_neg_{}".format(words['rank'])),  index_test_neg)

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5): #was 5
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = build_feature_vector("train_pos_{}", len(train_pos), model)
    train_neg_vec = build_feature_vector("train_neg_{}", len(train_neg), model)
    test_pos_vec =  build_feature_vector("test_pos_{}", len(test_pos), model)
    test_neg_vec =  build_feature_vector("test_neg_{}", len(test_neg), model)

    #print train_pos_vec

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    X = np.array(train_pos_vec + train_neg_vec, dtype=object)

    nb_model = BernoulliNB( alpha = 1.0, binarize = None )
    nb_model.fit(X, Y)

    lr_model = LogisticRegression()
    lr_model.fit(X, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = np.array(train_pos_vec + train_neg_vec, dtype=object)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    nb_model = GaussianNB()
    nb_model.fit(X, Y)

    lr_model = LogisticRegression()
    lr_model.fit(X, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    predicted_pos = model.predict(test_pos_vec)
    predicted_neg = model.predict(test_neg_vec)

    tp = len([v for (v) in predicted_pos if v == 'pos'])
    fn = len([v for (v) in predicted_pos if v != 'pos'])
    tn = len([v for (v) in predicted_neg if v == 'neg'])
    fp = len([v for (v) in predicted_neg if v != 'neg'])

    #print tp, fn, tn, fp

    accuracy = float(tn+tp)/float(tp+fn+tn+fp)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()
