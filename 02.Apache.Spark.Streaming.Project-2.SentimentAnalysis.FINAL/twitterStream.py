from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import util
import operator
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import re

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)#originally 10   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100) #originally 100
    make_plot(counts)

def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    x_dim = []
    y_dim_pos = []
    y_dim_neg = []

    for idx, count in enumerate(counts):
        x_dim.append(idx)

        if len(count) > 0:
            for tup in count:
                if( tup[0] == "positive"):
                    y_dim_pos.append(tup[1])
                if( tup[0] == "negative"):
                    y_dim_neg.append(tup[1])
        else:
            y_dim_pos.append(0)
            y_dim_neg.append(0)

    pos_line, = plt.plot(x_dim, y_dim_pos, "bo-")
    neg_line, = plt.plot(x_dim, y_dim_neg, "go-")
    plt.xlabel("Time step")
    plt.ylabel("Word count")

    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - 0.5,
              x1 + 0.5,
              y0 - 20,
              y1 + 50))

    plt.legend([pos_line, neg_line], ['positive', 'negative'], loc=2)

    plt.show()


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    words = set()
    for str in util.read_file(filename):
        words.add(str)

    return words


def stream(ssc, pwords, nwords, duration):
    pp = pprint.PrettyPrinter(indent=4)
    params = {"metadata.broker.list": 'localhost:9092'}

    kstream = KafkaUtils.createDirectStream(ssc,
                                            topics = ['twitterstream'],
                                            kafkaParams = params)
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).


    def takeCounts(word):
        pos,neg = (0,)*2
        if word in pwords:
            pos += 1
        if word in nwords:
            neg += 1
        return [("positive",pos),("negative",neg)]

    punctuationRegex = r"[\\\'\!\"\#\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]"

    tweetCounts = tweets\
                   .flatMap(lambda wordList: wordList.split(" "))\
                   .map(lambda word: re.sub(punctuationRegex, "", word))\
                   .map(lambda word: word.lower())\
                   .flatMap(takeCounts)\
                   .reduceByKey(lambda cnt,cnt2: cnt+cnt2)\

    tweetCounts.pprint()

    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    tweetCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()

