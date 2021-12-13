# housing logic here for approximate k-nn using hierarchal small worlds structure
# algorithms provided by TA from:
# https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf

import dataStructures as ds

# gets neighborhood of a node in a given layer
# v is the vertex to get the neighborhood of (v has an index)
# G is the multi layer graph
# layerNumber is the layer to consider in neighbor-grabbing
def getNeighborhood(v, G, layerNumber):
    neighbors = []
    startNode = G.layers[layerNumber][v]
    nextNode = startNode.next
    while nextNode is not None:
        neighbors.append(nextNode)
        nextNode = nextNode.next
    # returning vertex numbers of the neighbors in the layer
    return neighbors

# simple method of getting neighbors
# e is the actual element tuple for comparison
# C is a list of other element tuples that represents a candidate list
# M is the number of neighbors to grab
def selectNeighborSimple(e, C, M, distanceFunction):
    # initializing a max heap of size M
    Q = ds.maxHeap()
    for candidate in C:
        # calculating the distance
        D = distanceFunction(e, candidate)
        if Q.size()<M:
            Q.insert((candidate,D))
        else:
            if Q.peekMax()[1] > D:
                Q.extractMax()
                Q.insert((candidate,D))
    # returning the queue of up to M neighbors from the candidate list C
    return Q

# heuristic method of getting neighbors
def selectNeighborsHeuristic(e, C, M, layerNumber, keepPrunedConnections, extendCandidates = False):
    # implement
    pass



