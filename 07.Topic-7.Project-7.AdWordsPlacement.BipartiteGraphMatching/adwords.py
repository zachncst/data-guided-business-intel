import sys
import collections
import random
import math
from operator import attrgetter

# Author: Zachary Taylor (zctaylor@ncsu.edu)

if len(sys.argv) != 2:
    print "python adwords.py <greedy|msvv|balance>"
    exit(1)
method = sys.argv[1]
random.seed(0)

class KeywordBid:
    def __init__(self, keyword, bid):
        self.keyword = keyword
        self.bid = float(bid)

class Advertiser:
    def __init__(self, _id, budget):
        self._id = _id
        self.keyword_bids = []
        self.budget = float(budget)
        self.start_budget = float(budget)

    def add_query(self, keyword, bid):
        self.keyword_bids.append(KeywordBid(keyword, bid))

    def make_bid(self, bid_amount):
        self.budget=self.budget-bid_amount
        return bid_amount

    def has_budget(self, bid_amount):
        if self.budget - bid_amount >= 0:
            return True
        else:
            return False
    def resetBudget(self):
        self.budget=self.start_budget

class AdvertiserBid:
    def __init__(self, _id, advertiser, bid):
        self._id = _id
        self.advertiser = advertiser
        self.bid = bid

    def make_bid(self):
        return self.advertiser.make_bid(self.bid)

    def has_budget(self):
        return self.advertiser.has_budget(self.bid)

    def print_str(self):
        return "Advertiser {0}, Bid {1}, Budget {2}".format(self._id, self.bid, self.advertiser.budget)

def make_advertiser_bid(ad, query):
    return AdvertiserBid(ad._id, ad, query.bid)

def read_queries(queries_file):
    queries_list = []

    for line in queries_file:
        queries_list.append(line.strip())

    return queries_list

def read_advertisers(advertiser_file):
    advertiser_file.next() #drop header
    curr_obj = None
    advertisers = []

    for line in advertiser_file :
        cols = line.split(",")

        if(curr_obj is None or curr_obj._id != int(cols[0])):
            curr_obj = Advertiser(int(cols[0]), cols[3])
            advertisers.append(curr_obj)

        curr_obj.add_query(cols[1],float(cols[2]))

    return advertisers

def build_query_lookup(advertisers):
    query_map = dict()

    for ad in advertisers:
        for query in ad.keyword_bids:
            bid_el = make_advertiser_bid(ad,query)
            if query.keyword in query_map:
                query_map[query.keyword].append(bid_el)
            else:
                query_map[query.keyword] = [bid_el]

    return query_map

def read_file(filename) :
    f = open(filename, "r")
    for line in f:
        yield line

def greedy(query, neighbors):
    # Greedy function, take the highest bid and don't monitor budget
    sorted_neighbors = sorted(neighbors, key=lambda x: (-x.bid, x._id))

    if(len(sorted_neighbors) != 0):
        return sorted_neighbors[0].make_bid()

    return 0.0

def psi(budget_ratio):
    return 1-math.pow(math.e, budget_ratio-1)

def budget_lens(ad):
    return ad.advertiser.budget

def start_budget_lens(ad):
    return ad.advertiser.start_budget

def budget_ratio(ad):
    ratio = 1-budget_lens(ad)/start_budget_lens(ad)
    return ratio

def msvv(query, neighbors):
    # MSVV function, use a weighted budget value to pick the bid
    sorted_neighbors = sorted(neighbors, key=lambda x: (-x.bid*psi(budget_ratio(x)), x._id))

    if(len(sorted_neighbors) != 0):
        return sorted_neighbors[0].make_bid()

    return 0.0

def balance(query, neighbors):
    # Match to highest unspent budget 
    sorted_neighbors = sorted(neighbors, key=lambda x: (-x.advertiser.budget, x._id))

    if(len(sorted_neighbors) != 0):
        return sorted_neighbors[0].make_bid()

    return 0.0

def driver(function, queries, query_map):
    revenue = 0.0

    for query in queries:
        if query in query_map:
            advertisers = query_map[query]
            filtered_advertisers = filter(lambda ar: ar.has_budget(), advertisers)
            revenue += function(query, filtered_advertisers)

    return revenue


def main():
    queries = read_queries(read_file("./queries.txt"))
    advertisers = read_advertisers(read_file("./bidder_dataset.csv"))
    query_map = build_query_lookup(advertisers)
    method_func = None

    if method.lower() == "greedy":
        method_func = greedy
    if method.lower() == "msvv":
        method_func = msvv
    if method.lower() == "balance":
        method_func = balance

    if method_func is None:
        print "Method " + method.lower() + " function not found"
        exit(1)

    opt_revenue = reduce(lambda x,y: x+y,map(lambda ad: ad.budget, advertisers))
    revenues = []

    for i in range(0,100):
        random.shuffle(queries)
        revenues.append(driver(method_func, queries, query_map))
        map(lambda x: x.resetBudget(), advertisers)

    revenue_mean = reduce(lambda x,y : x+y, revenues)/len(revenues)
    revenue_ratio = revenue_mean/opt_revenue

    print "algorithm: {}\trevenue\t{:>.2f}\tcompetitive ratio\t{:>.2f}".format(method, revenue_mean, revenue_ratio)


if __name__ == "__main__":
    main()
