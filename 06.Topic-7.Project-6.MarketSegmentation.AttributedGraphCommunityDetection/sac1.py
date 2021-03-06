import sys
import igraph
import func_utils as func
import copy
import math
from collections import defaultdict
from itertools import takewhile, starmap
from operator import itemgetter

#Author Zachary Taylor (zctaylor@ncsu.edu)
def read_file(filename) :
    f = open(filename, "r")
    for line in f:
        yield line


def make_attr_map(in_file):
    attr_map = {}
    i = 0

    keys = in_file.next().split(",")

    for line in in_file:
        values = map(lambda x: int(x), line.split(","))
        attr_map[int(i)] = values
        i += 1

    return attr_map

if len(sys.argv) != 2:
    print "python sac1.py <alpha>"
    exit(1)
alpha = float(sys.argv[1])
attribute_map = make_attr_map(read_file('./data/fb_caltech_small_attrlist.csv'))


def compare(attr1, attr2):
    result = func.pipe(
        (map, lambda x: 1 if x[0] == x[1] else 0),
        (reduce, lambda x,y: x+y)
    )(zip(attr1,attr2))

    return float(result)

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def simularity(graph, indices):
    attributes = []

    for i in indices:
        attr1 = graph.vs[i]["sim"]

        if isinstance(attr1[0], list):
            attributes.extend(attr1)
        else :
            attributes.append(attr1)

    results = []
    for i in range(0, len(attributes)):
        for j in range(i+1, len(attributes)):
            results.append(cosine_similarity(attributes[i], attributes[j]))

    if len(results) == 0:
        return 0
    return reduce(lambda x,y: x+y, results)/float(len(results))

def sum_attributes(x):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    result = x

    if len(x) > 0:
        if len(x[0]) > 0 and isinstance(x[0][0], list):
            result = [item for sublist in x for item in sublist]


    return result

def sac1(graph):
    results = []

    attributes = [attribute_map[x] for i, x in enumerate(attribute_map.keys())]
    weights = [1 for x in range(0, graph.ecount())]

    graph.es["weight"] = weights
    graph.vs["sim"] = attributes
    #graph.vs["community"] = []

    for k in range(0, 15):
        membership = [(x) for x in range(0, graph.vcount())]
        membership_old = copy.copy(membership)
        clustering_old = igraph.VertexClustering(graph, membership)
        #igraph.plot(clustering_old)

        print igraph.summary(clustering_old)

        #A pass
        for k in range(0, 15):
            starting_membership = copy.copy(membership)

            for vert in range(0, len(membership)):
                mod_results = []
                q_newman_cached = {}
                community_size = len(set(membership))

                vert_old = igraph.VertexClustering(graph, membership=membership)
                mod_old = vert_old.modularity

                for vertj in range(0, len(membership)):
                    community = membership[vertj]

                    if community not in q_newman_cached:
                        membership_copy = copy.copy(membership)
                        membership_copy[vert] = community
                        community_size_new = len(set(membership_copy))
                        comm_indices = [i for i, x in enumerate(membership) if x == community]
                        comm_indices_new = [i for i, x in enumerate(membership_copy) if x == community]

                        vert_new = igraph.VertexClustering(graph, membership=membership_copy)
                        mod_new = vert_new.modularity

                        modularity_diff = mod_new - mod_old

                        #if modularity_diff > 0:
                            #print "Modularity", modularity_new, "-", modularity_old, "=", modularity_diff
                        #print "Mod       ", mod_new, "-", mod_old, "=", modularity_diff

                        sim_result_old = simularity(graph, comm_indices)
                        sim_result_new = simularity(graph, comm_indices_new)

                        #print sim_result_old, sim_result_new

                        sim_result = (sim_result_new - sim_result_old)
                        q_newman = alpha*modularity_diff + (1-alpha)*(sim_result)/(math.pow(community_size_new, 2))
                        q_newman_cached[community] = q_newman
                        result = (community, q_newman)
                        mod_results.append(result)

                filtered_results = filter(lambda (c,m): m > 0, mod_results)

                if len(filtered_results) > 0:
                    sorted_results = sorted(filtered_results, key=itemgetter(1), reverse=True)
                    membership[vert] = sorted_results[0][0]

            diff = reduce(lambda x,y: x+y, map(lambda (x,y): 1 if x != y else 0, zip(starting_membership, membership)), 0)
            print "Membership diff of", diff

            if starting_membership == membership:
                print "No further changes can be made"
                break;

        if len(results) != 0 and results[len(results)-1]== membership:
            print "No further improvements, finished on ", k
            break;

        previous_communities = None
        if "community" in set(graph.vertex_attributes()):
            previous_communities = {i:e for i,e in enumerate(graph.vs["community"])}
            #print previous_communities

        results.append(copy.copy(membership))
        optimal_membership = copy.copy(membership)

        #Rename optimal membership so it'll remove nodes, communities should be 0 to n.
        for k, x in enumerate(sorted(set(optimal_membership))):
            for l, y in enumerate(optimal_membership):
                if x == y:
                    optimal_membership[l] = k

        print optimal_membership
        combinations = {
            "sim" : lambda x: sum_attributes(x)
        }
        graph.contract_vertices(optimal_membership, combine_attrs=combinations)
        
        community_dict = defaultdict(list)

        for k, x in enumerate(optimal_membership):
            community_dict[x].append(k)

        if previous_communities is None :
            community_list = [set(community_dict[l]) for l in community_dict]
        else :
            community_list = [[previous_communities[c] for c in community_dict[l]] for l in community_dict]
            community_list = map(lambda x: [item for sublist in x for item in sublist], community_list)
            print community_list

        graph.vs["community"] = community_list
        graph.simplify(combine_edges=dict(weight="sum"), multiple=True, loops=False)

    return graph.vs["community"]

    
def main():
    facebook_graph = igraph.load('./data/fb_caltech_small_edgelist.txt')
    final_communities = sac1(facebook_graph)
    file = open('communities.txt', 'w+')

    for c in final_communities:
        community = map(lambda x: str(x), c)
        file.write(", ".join(community) + "\n")

    file.close()

if __name__ == "__main__":
    main()

