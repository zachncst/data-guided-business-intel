import sys
import collections

if len(sys.argv) != 2:
    print "python adwords.py <greedy|msvv|balance>"
    exit(1)
method = sys.argv[1]

class KeywordBid:
    def __init__(self, keyword, bid):
        self.keyword = keyword
        self.bid = bid

class Advertiser:
    def __init__(self, _id, budget):
        self._id = _id
        self.keyword_bids = []
        self.budget = budget

    def add_query(self, keyword, bid):
        self.keyword_bids.append(KeywordBid(keyword, bid))



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


def read_file(filename) :
    f = open(filename, "r")
    for line in f:
        yield line


def greedy():
    # Greedy function placeholder
    print "hold"

def msvv():
    # MSVV function placeholder
    print "hold"

def balance():
    # Balance function placeholder
    print "hold"


def main():
    queries = read_file("./queries.txt")
    advertisers = read_advertisers(read_file("./bidder_dataset.csv"))
    method_func = None

    if method.lower() == "greedy":
        method_func = greedy
    if method.lower() == "msvv":
        method_func = mssv
    if method.lower() == "balance":
        method_func = balance

    if method_func is None:
        print "Method " + method.lower() + " function not found"
        exit(1)

    print "Using method " + method.lower()



if __name__ == "__main__":
    main()
