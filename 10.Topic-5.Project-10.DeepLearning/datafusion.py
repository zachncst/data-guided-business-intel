#===============================================================
# Name: Data Fusion Project
# Vers: 1.0
# author : Zachary Taylor (zctaylor@ncsu.edu)
# Description: In this project, we will have you
# implement the algorithm discussed in "paper citation"
# using Tensorflow to solve the issue of image classification. 
# In addition, we will also ask you to clean and filter the 
# training data to reduce training time. At the end of the 
# project, the following learning goals should be achieved:
#   1. How to take data of different modalities and combine
#       them to answer a question about the entire dataset
#   2. How to sample and reduce very large datasets to test
#       and validate your algorithm
#===============================================================

import sys
import os
import string
import math
from itertools import islice
import collections as cl
import numpy as np
import numpy.random as npr
import tensorflow as tf
import networkx as nx
from collections import defaultdict
#import rawdatacleaner as rdc

def load_textcorpus():
    """
        Reads the bbc datafile and generates a dictionary, where the keys
        are the text categories, and the values are a list of the documents,
        with each document being just being the multiset of words in the document
    """
    textfiles = []
    datafile = "data/bbc"
    for l in filter(lambda x: "." not in x, os.listdir(datafile)):
        for f in os.listdir(datafile+os.sep+l):
            with open(datafile+os.sep+l+os.sep+f, "r") as of:
                document = []
                for line in of:
                    document.extend(line.lower().strip().strip(string.punctuation).split())
                textfiles.append(document)
    return textfiles

def load_imagecorpus():
    """
        Reads the saved numpy matrices that store 5 127-by-127 patches for each image in
        the NUS-WIDE dateset that contain a certain keyword (discussed in the project description)

        Also reads the image_tag file and generates the tags related to the image
    """
    matrix_dir = "data/Mats"
    image_tag_file = "data/imagelist.txt"

    images = {}
    tags = {}
    image_tag_file = open(image_tag_file, "r")
    for irow in image_tag_file:
        row = irow.split()
        iname = row[0]
        if len(row[1:] ) > 0:
            tags[iname] = row[1:]
            try :
                images[iname] = np.load(os.path.join(matrix_dir, iname.split("\\")[0]+os.sep+iname.split("\\")[1].split(".")[0])+".npy")
            except:
                #print "Unexpected error:", sys.exc_info()[1]
                pass

    return (tags,images)

def build_tag_vocab(tags):
    """
        This file generates the one-hot-encoding (binary) representation
        of each tag in the image's tag corpus, and return it as a dictionary
        where the keys are the tags, and the values are their binary representation
        (as a numpy array of integers)
    """
    ohe_tags = {}
    vocab_size = len(tags)
    for (i,w) in enumerate(tags):
        ohe_tags[w] = i

    return ohe_tags

def build_text_vocab(textfiles):
    """
        This file generates the one-hot-encoding (binary) representation
        of each tag in the image's tag corpus, and return it as a dictionary
        where the keys are the tags, and the values are their binary representation
        (as a numpy array of integers)
    """
    ohe_textfiles = {}
    vocab_size = len(textfiles)
    for (i,w) in enumerate(textfiles):
        ohe_textfiles[w] = i
    return ohe_textfiles

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def build_image_inputouput_set(tags, images, ohe_tags):
    """
        For each image in the training set, generate
        a tuple whose first position is a single 127-by-127
        patch, the the second position is a list of the indices
        representing the tags associated with the patch
    """
    image_inputoutput_set = []

    for key in images.keys():
        zero_array = [0]*len(tags.keys())
        tag_array = map(lambda x: ohe_tags[x], tags[key])

        for tag in tag_array:
            zero_array[tag] = 1

        image_array = images[key]

        try :
            for patch in image_array:
                if len(patch) != 127:
                    raise
                image_inputoutput_set.append((patch, zero_array))
        except:
            #print "key", key, "Unexpected error:", sys.exc_info()[1]
            pass

    return image_inputoutput_set

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def build_textcorpus_inputoutput_set(textcorpus, ohe_textfiles):
    """
        Build the input/output label pairs by applying a sliding
        window over the text corpus, and extracting the center
        and context words, as dicussed in the lectures.

        Set the window size to 5
        Let the input be a list containing the index of the center element
        Let the output be a list containing the indices of the context elements
    """
    window_size = 5
    textcorpus_inputoutput_set = []

    for corpus in textcorpus:
        for win in window(corpus, n=window_size):
            left = ohe_textfiles[win[2]]
            right = map(lambda i: ohe_textfiles[win[i]], filter(lambda x: x != 2, [i for i in range(5)]))
            textcorpus_inputoutput_set.append((left, right))

    return textcorpus_inputoutput_set

def build_skipgram(text_input, text_output, vocabulary_size):
    """
        Implement the skipgram algorithm. Please read the tensorflow tutorial
        on skipgram, which will walk you through how to do this in tensorflow
        https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words

        Set the number of hidden dimensions to 300

        Return the optimization operation, the loss and the embedding weight for text_input
    """
    embedding_size = 300
    num_sampled = 100

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    embedding = tf.nn.embedding_lookup(embeddings, text_input)

    #Computer the NCE Loss, using a sample of the negative labels each time
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embedding, text_output, num_sampled, vocabulary_size))

    #we use the SGD optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    return (loss, optimizer, embedding)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def build_cnn(image_input, image_output, tag_size):
    """
        Implement a cnn to embed the images in relation to their tags. For a walkthough
        on how to build cnns, please refer to this tensorflow tutorial:
        https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

        For the purposes of this project, we want a CNN with one convolutional
        layers and two softmax layers, with the second serving as the output layer.

        For Convolution 1, use a 5-by-5 filter with a stride of 2. Adjust the number of convolutional filters to 32

        Set the embedding weight vector equal to the output of the first convolutional layer (without drop-out)
    """
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_input, [-1, 127, 127, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_pool1_flat = tf.reshape(h_pool1, [-1, 32])

    W_fc1 = weight_variable([32, 1024])
    b_fc1 = bias_variable([1024])

    y_conv1 = tf.nn.softmax(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    cnn_embedding = y_conv1

    W_fc2 = weight_variable([1024, tag_size])
    b_fc2 = bias_variable([tag_size])
    y_conv2 = tf.nn.softmax(tf.matmul(y_conv1, W_fc2) + b_fc2)

    #Include this as part of code.
    loss = -tf.reduce_sum(image_output*tf.log(y_conv2))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return (loss, optimizer, cnn_embedding)

def build_adjacency_graph(tags, textcorpus):
    """
        Build the adjacency graph used by the heterogeneous data fusion algorithm to determine
        if an image and a word should be embedding close to one another. An image and an word should
        have an edge between them if the word is contained in the image's tag list

        A good graph library:
        http://networkx.readthedocs.org/en/networkx-1.11/

        Hint1: Instead of using the images as nodes, you can give your nodes attributes to denote
        "image nodes" and "text nodes"
    """
    adjacency_graph = nx.Graph()

    # print "tags", take(1, tags.values())
    # print "textcorpus", take(1, textcorpus)

    tags_to_image = defaultdict(list)

    for image_key in tags.keys():
        adjacency_graph.add_node(image_key, type='image')

        for tg in tags[image_key]:
            tags_to_image[tg].append(image_key)

    for corpus in textcorpus:
        for word in corpus:
            adjacency_graph.add_node(word, type='text')

            if word in tags_to_image:
                for tg in tags_to_image[word]:
                    adjacency_graph.add_edge(tg, word)

    return adjacency_graph

def build_noise_list(agraph):
    """
        Using the adjaceny graph, generate a list of 1000 pairs of non-adjacent images and words.
        This will be added to the learning set to make sure our algorithm can distinguish between
        words and images that should not be close.
    """

    noise_list = []

    for node in nodes(agraph):
        for non_neighbor in nx.non_neighbors(agraph, node):
            if node[type] != non_neight[type]:
                noise_list.append((node, non_neighbor))

            if len(noise_list) >= 1000:
                break

        if len(noise_list) >= 1000:
                break

    return noise_list

def data_fusion_loss_function(pdi, pdt, edge_exists):
    """
        This function should implement the loss function described by formula 18 in the paper
    """

    # Compute the dot-product of pdi and pdt and multiply by the negated edge_exists (-1 if 1, 1 if negative 1)

    dotprod = tf.Variable(-edge_exists*tf.matmul(tf.transpose(pdi), pdt))
    loss_function = tf.log(1 + tf.exp(dotprod))

    return (dotprod, loss_function);


def build_data_fusion_layer(t_embedding, i_embedding, edge_exists):
    """
        Using the lecture notes and your previous word2vec implementation, implement the data fusion
        layer described by formula 16 from the paper that will take in the embeddings between an 
        image embedding and a text embedding and compute the embedding and loss for the embedding.

        Note: Because we are only optimizing the embedding with repect to images and text, and not text-to-text
        or image-to-image, the objective function simplfies to just the loss function of the image to text portion

        Set the embedding dimension for this new projection layer to be 150
    """
    # Incorporating dropout on the hidden layer:
    dropped_tembedding = tf.nn.dropout(t_embedding, keep_prob=.5)
    dropped_iembedding = tf.nn.dropout(i_embedding, keep_prob=.5)

    # Create the variables associated with the transformation weights
    #Using fromula 16 from the paper
    d = 150

    #bt = tf.constant(1)
    #embeddings = tf.Variable(tf.random_uniform([d, 1024], -1.0, 1.0))

    #def qdt(z) :
    #    return tf.maximum(0, tf.matmul(embeddings, z) + bt)

    #pdi = qdt(dropped_tembedding)
    #pdt = qdt(dropped_tembedding)

    Ut = tf.Variable(tf.random_uniform([d, 1024], -1.0, 1.0))
    Uz = tf.Variable(tf.random_uniform([d, 1024], -1.0, 1.0))

    # Transform the embeddings with Uz and Ut
    pdi = tf.matmul(Ut, dropped_iembedding)
    pdt = tf.matmul(Uz, dropped_tembedding)

    (dropprod, loss_function) = data_fusion_loss_function(pdi, pdt, edge_exists)
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_function)

    return (optimizer, loss_function, Ut, Uz, dropprod, pdi, pdt)


def query(sess, pdi, pdt, image, frequent_words):
    """
        Given an image, retreive the most frequent words that might be strongly
        associated with the image using the vectors generated by the heterogeneous
        data fusion method
    """
    words = []

    for word in frequent_words:
        _,_,_,_, drop = sess.run([doptimizer, dloss], {image_input:image, \
            image_truth: pdi, text_input:batch[1], text_truth: pdt, edge_exists:[1]})

        if drop > 0:
            words.append(word)

    return words

def main():
    print "running main"
    # Number of iterations for the heterogeneous data fusion algorithm to train
    num_iters = 5

    text = load_textcorpus()
    (image_tags, images) = load_imagecorpus()

    tags = set()
    count = cl.Counter()

    #Add all words from text corpus
    for l in text:
        count.update(l)

    words = count.keys()
    # print words

    vocabulary_size = len(words)
    tag_size = len(image_tags)
    print vocabulary_size, '   vocab size'
    freq_words = count.most_common(1000)

    for t in image_tags.itervalues():
        tags.update(t)

    ohe_text = build_text_vocab(words)
    ohe_tags = build_tag_vocab(tags)

    text_learning = build_textcorpus_inputoutput_set(text, ohe_text)
    image_learning = build_image_inputouput_set(image_tags, images, ohe_tags)
    print len(image_learning), '    image size'

    relationgraph = build_adjacency_graph(image_tags, text)

    with tf.Graph().as_default() as g:
        text_input = tf.placeholder("int32",shape=[None])
        text_truth = tf.placeholder("float",shape=[None, None])

        image_input = tf.placeholder("float",shape=[127,127])
        blank_image = np.zeros(127)
        image_truth = tf.placeholder("float",shape=[None, None])

        edge_exists = tf.placeholder("float",shape=[1])

        with tf.name_scope("skipgram"):
            (toptimizer, tloss, tembedding) = build_skipgram(text_input, text_truth, vocabulary_size)

        with tf.name_scope("cnn"):
            (ioptimizer, iloss, iembedding) = build_cnn(image_input, image_truth, tag_size)

        with tf.name_scope("data_fusion_layer"):
            (doptimizer, dloss, Ut, Uz, dropprod, pdi, pdt) = build_data_fusion_layer(tembedding, iembedding, edge_exists)

        with tf.Session(graph=g) as sess:
            print "Start initialize"
            init_op = tf.initialize_all_variables()
            try :
                sess.run(init_op)
            except:
                pass

            print "Start image_learning"
            i = 0

            #with tf.name_scope("cnn"):
            for batch in image_learning:
                i += 1
                if i % 100 == 0:
                    print i, "iteration of", len(image_learning)
                
                _,loss = sess.run([ioptimizer, iloss],{image_input:batch[0], \
                    image_truth:[batch[1]], text_input:[], text_truth:[[]], edge_exists:[-1]})

            # Now, train the skipgram neural network. Look to the previous for loop
            # for suggestions on how to perform this
            print "Stop image_learning"

            i = 0

            print "Start text_learning"

            for batch in text_learning:
                i += 1
                if i % 100 == 0:
                    print i, "iteration of", len(text_learning)
                _,loss = sess.run([toptimizer, tloss],{image_input:[[]], \
                    image_truth:[], text_input:batch[0], text_truth:batch[1], edge_exists:[-1]})

            print "Stop text_learning"

            for i in range(0,num_iters):
                print "before noise list"
                noise_list = build_noise_list(relationgraph)
                print "after noise list"

                # Normally you would permute this. However, due to the large size of each list,
                # this permuation step would dominate training.
                i = 0

                print "before fusion 1"
                for batch in relationgraph.edges():
                    i += 1
                    if i % 100 == 0:
                        print i, "iteration of", len(relationgraph.edges())

                    _,loss = sess.run([doptimizer, dloss], {image_input:batch[0], \
                        image_truth:[], text_input:batch[1], text_truth:[], edge_exists:[1]})

                print "after fusion 1"
                # Now, train the data fusion layer by passing each image_word pair that should not be adjacent in the noise_list
                # by passing the tuples one at a time, with edge_exists equal to -1
                print "before fusion 2"
_
                for batch in noise_list:
                    i += 1
                    if i % 100 == 0:
                        print i, "iteration of", len(noise_list)

                    _,loss = sess.run([doptimizer, dloss], {image_input:batch[0], \
                        image_truth:[], text_input:batch[1], text_truth:[], edge_exists:[-1]})

                print "after fusion 2"

            edges_predicted = 0
            for batch in relationgraph.edges():
                # For each batch, get the value of the dropprod computed by the computational graph
                # If this value is above zero, increment edges_predicted by one
                _,_,_,_, drop = sess.run([doptimizer, dloss], {image_input:batch[0], \
                        image_truth:[], text_input:batch[1], text_truth:[], edge_exists:[1]})

                if drop > 0:
                    edges_predicted += 1

            print "--------------------------------"
            print "Reconstruction Error: %d" % (relationgraph.num_of_edges/edges_predicts)
            print "--------------------------------"
            print "Image 100 Tags"
            print image_tags[image[100]]
            print query(sess, pdi, pdt, images[100], image_tags[images[100]], freq_words)
            print zip([1,0],["a","b","c"])

if __name__ == "__main__":
    main()
