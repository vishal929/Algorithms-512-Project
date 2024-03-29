# python file that houses our defined data structures and some helper functions
# data structures implemented:
# min/max heap
# KD tree
# multilayer graph with hierarchical navigable small worlds functions

from copy import deepcopy
from math import log
from random import random

# helper function for selection
# selects kth smallest
def select(dataset,i,j,featureForSelection, k):
    pass

# helper function to run insertion sort on part of an array based on a specific feature
def insertionSortFeature(dataset, i,j, featureToUse):
    # sorts the array based on the feature using insertion sort
    '''
    print("presorted on feature: " + str(featureToUse))
    print(dataset[i:j+1])
    '''
    if i==j:
        return
    for k in range(i+1,j+1):
        tuple = dataset[k].copy()
        #print("erai" + str(tuple))
        value = tuple[featureToUse]
        lastIndex = k-1
        while lastIndex >=i and value < dataset[lastIndex][featureToUse]:
            # swapping
            dataset[lastIndex+1] = dataset[lastIndex]
            #print("datasetSwap")
            #print(dataset[i:j+1])
            lastIndex -= 1
        # setting the correct value index
        #print(tuple)
        dataset[lastIndex+1] = tuple
        #print("done inserting")
        #print(dataset[i:j+1])
    # array is sorted in place
    #print("postsorted on feature: " + str(featureToUse))
    #print(dataset[i:j + 1])



# helper function to get the median of medians of the portion of the dataset based on a specific feature
def medianFeature(dataArray,i,j,featureToUse):
    if j-i+1 <= 5:
        # we will just call insertion sort and return the median position
        insertionSortFeature(dataArray,i,j,featureToUse)
        return (i+j)/2
    # else, we can split the input into subarray of at most 5 elements, compute the median of those
    # then recursively call to grab the median of the medians
    currList = []
    medianList = []
    numAppended = 0
    for k in range(i,j+1):
        currList.append(dataArray[k])
        numAppended += 1
        if numAppended % 5 == 0 :
            # then we compute the median and append it to the median list
            insertionSortFeature(currList,0,len(currList)-1,featureToUse)
            # appending the median
            medianList.append(currList[int((len(currList)-1)/2)])
            # resetting the current list
            currList = []
    # might have incomplete list here
    if len(currList)!=0:
        # sorting list of size at most 5
        insertionSortFeature(currList, 0, len(currList) - 1, featureToUse)
        # appending the median
        medianList.append(currList[int((len(currList) - 1) / 2)])
        # resetting the current list
        currList = []
    # now we have a list of medians of medians







# helper function to pivot a dataset based on a certain feature value
# pivots the array about a partition point -> does it in place
def partitionFeature(dataset, i,j, featureToPivot, indexToPivot):
    if indexToPivot != j:
        # then we can swap them so we always partition about the last index
        temp = dataset[j]
        dataset[j] = dataset[indexToPivot]
        dataset[indexToPivot] = temp
    pivotValue = dataset[indexToPivot][featureToPivot]
    z = i -1
    for k in range(i,j):
        if dataset[k][featureToPivot] <= pivotValue:
           z += 1

    z += 1
    temp = dataset[j]
    dataset[j] = dataset[z]
    dataset[z] = temp
    # returning the index where the pivot was placed
    return z

class minHeap(object):
    def __init__(self):
        self.arr = []
    def size(self):
        return len(self.arr)
    def isEmpty(self):
        return len(self.arr) == 0
    # just a function to peek the maximum element
    def peekMin(self):
        if self.isEmpty():
            return None
        else:
            return self.arr[0]
    # function to sift up (sifting up from the given index)
    def siftUp(self, index):
        # getting the parent, doing a comparison
        # if this element is greater than the parent, we have to swap and repeat the sift
        # using an array for recursion based on stack, instead of recursive calls
        siftStack = [index]
        while len(siftStack)>0:
            index = siftStack.pop()
            if index == 0:
                # then we are at the root, cant sift up anymore
                continue
            # getting parent index
            parent = int((index-1)/2)
            if self.arr[parent][1] > self.arr[index][1]:
                # then we have to swap and continue the sift up
                temp = self.arr[parent]
                self.arr[parent] = self.arr[index]
                self.arr[index] = temp
                siftStack.append(parent)
    # function to sift down
    def siftDown(self, index):
        # we get the children of the current node and perform a comparison
        # we will use a stack for recursion instead of recursive calls
        # note that if the current element is a leaf, then there is no need to sift down any further
        siftStack = [index]
        while len(siftStack)>0:
            siftDownIndex = siftStack.pop()
            leftChild = (2*siftDownIndex)+1
            rightChild = leftChild +1
            # since this is a min heap, children should be >= parent
            minimumIndex = siftDownIndex
            if leftChild is not None and leftChild < len(self.arr) and self.arr[leftChild][1] < self.arr[minimumIndex][1]:
                minimumIndex = leftChild
            if rightChild is not None and rightChild < len(self.arr) and self.arr[rightChild][1] < self.arr[minimumIndex][1]:
                minimumIndex= rightChild
            if minimumIndex != siftDownIndex:
                # then we swap the elements, and continue the sift down
                temp = self.arr[siftDownIndex]
                self.arr[siftDownIndex] = self.arr[minimumIndex]
                self.arr[minimumIndex] = temp
                # setting up next iteration of "recursion"
                siftStack.append(minimumIndex)
    # function to insert into the queue
    # e is the element to insert
    # e should be formatted as the following tuple: (data, distance)
    def insert(self,e):
        # need to insert the element as a leaf and sift up
        self.arr.append(e)
        # sifting up
        self.siftUp(len(self.arr)-1)
    # function to extract a minimum element from the minHeap
    def extractMin(self):
        # return the root, and then replace the root value with the last leaf, and then sift down
        maxData = self.arr[0]
        if len(self.arr) == 1:
            self.arr.pop()
            return maxData
        # swapping root with leaf
        self.arr[0] = self.arr[len(self.arr)-1]
        # removing the leaf
        #self.arr.remove(len(self.arr)-1)
        self.arr.pop(len(self.arr)-1)
        # sifting down
        self.siftDown(0)
        return maxData
    # function to build the heap from a specific array (linear time)
    def heapify(self, array):
        self.arr = array
        # going from the back of the array and calling sift down
        for i in range(len(self.arr) - 1, -1, -1):
            self.siftDown(i)

class maxHeap(object):
    def __init__(self):
        self.arr = []
    def size(self):
        return len(self.arr)
    def isEmpty(self):
        return len(self.arr) == 0
    # just a function to peek the maximum element
    def peekMax(self):
        if self.isEmpty():
            return None
        else:
            return self.arr[0]
    # function to sift up (sifting up from the given index)
    def siftUp(self, index):
        # getting the parent, doing a comparison
        # if this element is greater than the parent, we have to swap and repeat the sift
        # using an array for recursion based on stack, instead of recursive calls
        siftStack = [index]
        while len(siftStack)>0:
            index = siftStack.pop()
            if index == 0:
                # then we are at the root, cant sift up anymore
                continue
            # getting parent index
            parent = int((index-1)/2)
            if self.arr[parent][1] < self.arr[index][1]:
                # then we have to swap and continue the sift up
                temp = self.arr[parent]
                self.arr[parent] = self.arr[index]
                self.arr[index] = temp
                siftStack.append(parent)
    # function to sift down
    def siftDown(self, index):
        # we get the children of the current node and perform a comparison
        # we will use a stack for recursion instead of recursive calls
        # note that if the current element is a leaf, then there is no need to sift down any further
        siftStack = [index]
        while len(siftStack)>0:
            siftDownIndex = siftStack.pop()
            leftChild = (2*siftDownIndex)+1
            rightChild = leftChild +1
            # since this is a max heap, children should be <= parent
            maximumIndex = siftDownIndex
            if leftChild is not None and leftChild < len(self.arr) and self.arr[leftChild][1] > self.arr[maximumIndex][1]:
                maximumIndex = leftChild
            if rightChild is not None and rightChild < len(self.arr) and self.arr[rightChild][1] > self.arr[maximumIndex][1]:
                maximumIndex = rightChild
            if maximumIndex != siftDownIndex:
                # then we swap the elements, and continue the sift down
                temp = self.arr[siftDownIndex]
                self.arr[siftDownIndex] = self.arr[maximumIndex]
                self.arr[maximumIndex] = temp
                # setting up next iteration of "recursion"
                siftStack.append(maximumIndex)
    # function to insert into the queue
    # e is the element to insert
    # e should be formatted as the following tuple: (data, distance)
    def insert(self,e):
        # need to insert the element as a leaf and sift up
        self.arr.append(e)
        # sifting up
        self.siftUp(len(self.arr)-1)
    # function to extract a maximum element from the maxHeap
    def extractMax(self):
        # return the root, and then replace the root value with the last leaf, and then sift down
        maxData = self.arr[0]
        if len(self.arr) == 1:
            self.arr.pop()
            return maxData
        # swapping root value with a leaf value
        self.arr[0] = self.arr[len(self.arr)-1]
        # removing the leaf value
        # self.arr.remove(len(self.arr)-1)
        self.arr.pop(len(self.arr)-1)
        # sifting down
        self.siftDown(0)
        return maxData
    # function to build the heap from a specific array (linear time)
    def heapify(self, array):
        self.arr = array
        # going from the back of the array and calling sift down
        for i in range(len(self.arr)-1,-1,-1):
            self.siftDown(i)

# class for nodes in a kdtree
class kdNode(object):
    def __init__(self,data,splitAxis,left,right):
        # data is a tuple of features for each datapoint
        self.data = data
        # split axis is the feature on which this node was split
        self.splitAxis = splitAxis
        # node has a left and right child for the future splits
        self.left = left
        self.right = right

# class for kd trees datastructure
class kdTree(object):
    # kdTree is defined by a root node
    # construction is defined by some dataset
    # ASSUMES THAT FEATURES NOT IMPORTANT FOR SPLITTING ARE AT THE END OF THE DATA TUPLE!!!
    def __init__(self,data, numFeatures):
        self.dataset = data
        # getting the number of features in the dataset based on the length of a data tuple
        self.numFeatures = numFeatures
        # constructing the kdtree from data and setting the root node
        self.rootNode = self.constructTreeDriver()
    # printing the nodes in a tree
    def printTree(self, node):
        if node is None:
            return
        print(node.data)
        self.printTree(node.left)
        self.printTree(node.right)
    # recursive construction of a kdtree from a dataset
    # data is an array of element tuples where parts of the tuple are features
    # i and j are indices for recursion
    # featureToSplit is just the feature number to split the next level on (dimension)
    def constructTree(self,i,j,featureToSplit):
        # recursively constructing tree from the dataset for the kdtree


        if i == j:
            # then we only have 1 piece of data in this subtree
            #print("added element at index: " + str(i))
            return kdNode(self.dataset[i], featureToSplit, None, None)

        if j<i:
            # no more data
            return None

        # pivoting data based on which feature we are using now
        # REPLACE: For right now, just using insertion sort and picking the middle point
        insertionSortFeature(self.dataset,i,j,featureToSplit)
        mid = int((i+j)/2)
        # put logic below for selection of the mid value
        #mid = pivotFeature(self.dataset,i,j,featureToSplit)

        #print("added element at index " + str(mid))
        parent = kdNode(self.dataset[mid],featureToSplit, None, None)
        # incrementing the feature to split on for the next level
        featureToSplit += 1
        # resetting the feature split to "wrap around"
        if featureToSplit >= self.numFeatures:
            featureToSplit = 0
        parent.left = self.constructTree(i,mid-1,featureToSplit)
        parent.right = self.constructTree(mid+1,j,featureToSplit)

        # returning the parent
        return parent

    # driver to construct a tree recursively and set it as the root node
    # this is called during init with a dataset of tuples
    def constructTreeDriver(self):
        return self.constructTree(0,len(self.dataset)-1,0)

    # actual function for knn in kd trees
    # element is the basis element for comparison with kd trees
    def kdTreeKNN(self,element,k, distanceFunction):
        # initializing a max heap for storing our k best neighbors
        # max heap because we want to compare the worst element with a new element for insertion
        W = maxHeap()
        # finding leaf node where this element would be inserted
        self.kdTreeKNNHelper(element,k,W,self.rootNode, distanceFunction)
        # now we have performed the recursive search
        # returning the best neighbors of the element as a maxHeap
        return W

    def kdTreeKNNHelper(self,element,k,maxHeap, currentNode, distanceFunction):
        if currentNode is None:
            return
        # doing a comparison
        D = distanceFunction(element,currentNode.data)
        if maxHeap.size()<k or maxHeap.peekMax()[1] > D:
            if maxHeap.size() >=k:
                maxHeap.extractMax()
            maxHeap.insert((currentNode.data, D))
        if currentNode.left is None and currentNode.right is None:
            return
        # traversing to place where element would be inserted
        # because of the above if statement, at most one of the 2 subtrees can be None
        nodeToRecurse = None
        if element[currentNode.splitAxis] < currentNode.data[currentNode.splitAxis]:
           nodeToRecurse = currentNode.left
        else:
           nodeToRecurse = currentNode.right

        # traversing deeper in the tree
        self.kdTreeKNNHelper(element,k,maxHeap,nodeToRecurse,distanceFunction)

        # seeing if there might be anything better in the other subtree after recursion
        currLargestDistance = maxHeap.peekMax()[1]
        # reason we use the largest here is because we want k nearest neighbors, so if we can replace the worst neighbor in the heap, then we should
        if nodeToRecurse == currentNode.left:
            if element[currentNode.splitAxis] + currLargestDistance >= currentNode.data[currentNode.splitAxis]:
                # then we also search the right subtree for better neighbors
                self.kdTreeKNNHelper(element,k,maxHeap,currentNode.right,distanceFunction)
        else:
            if element[currentNode.splitAxis] - currLargestDistance <= currentNode.data[currentNode.splitAxis]:
                # then we can also search the left subtree for better neighbors
                self.kdTreeKNNHelper(element, k, maxHeap, currentNode.left, distanceFunction)






class adjacencyVertices(object):
    def __init__(self,data):
       self.data = data

class adjacencyNode(object):
    def __init__(self, vertexNumber):
        # vertex associated with the adjacency edge (vertex number)
        self.vertex = vertexNumber
        # edge to the next vertex in the adjacency list
        self.next = None


# class for the multilayer graph to be implemented as part of the hierarchal small world graphs implementation
class multiLayerGraph(object):
    def __init__(self):
        # basic idea in a small world graph is the following:
        # edges in each layer scale logarithmically and the final layer will have a neighborhood of closest neighbors (in theory)
        # having a list of possible vertices
        # layers in the graph, where each layer contains an adjacency list
        # each layer contains a single adjacency list for greedy search
        # layer 0 will contain the most edges, to nail down approximate knn
        # topmost layer will have an entry point with sparse edges
        # self.vertices is a list mapping vertex number to the tuple of data
        self.vertices = [] # array of
        # self.layers is a list of layers of adjacency lists for the hierarchal structure
        # adjacency lists have a vertex number
        self.layers = [] # array of arrays of type adjacencyNode
        # hierarchal structure has a fixed entry point to the top layer in the graph
        # this is changed during construction
        self.enterPoint = None
    # gets neighborhood of a node in a given layer
    # v is the vertex to get the neighborhood of (v has an index)
    # G is the multi layer graph
    # layerNumber is the layer to consider in neighbor-grabbing
    def getNeighborhood(self, v, layerNumber):
        neighbors = []
        startNode = self.layers[layerNumber][v]
        while startNode is not None:
            neighbors.append(startNode.vertex)
            startNode = startNode.next
        # returning vertex numbers of the neighbors in the layer
        return neighbors

    # simple method of getting neighbors
    # e is the actual element tuple for comparison
    # C is a list of other element tuples that represents a candidate list
    # M is the number of neighbors to grab
    def selectNeighborSimple(self, e, C, M, distanceFunction):
        # initializing a max heap of size M
        Q = maxHeap()
        for candidate in C:
            # candidate is just a vertex number
            # realCandidate is the data tuple
            realCandidate = self.vertices[candidate]
            # calculating the distance
            D = distanceFunction(e, realCandidate)
            if Q.size() < M:
                Q.insert((candidate, D))
            else:
                if Q.peekMax()[1] > D:
                    Q.extractMax()
                    Q.insert((candidate, D))
        # returning the maxHeap of up to M neighbors from the candidate list C
        return Q

    # heuristic method of getting neighbors
    def selectNeighborsHeuristic(self, distanceFunction, e, C, M, layerNumber, keepPrunedConnections, extendCandidates=False):
        # R is a max heap
        R = maxHeap()
        # min heap of candidate datapoints
        # assuming that C has input as an array, so we have to heapify into W with distances
        distanceArray = deepcopy(C)
        for i in range(len(distanceArray)):
            distanceArray[i] = (distanceArray[i], distanceFunction(e,self.vertices[distanceArray[i]]))

        W = minHeap()
        W.heapify(distanceArray)
        if extendCandidates:
            for candidate in C:
                # candidate is the vertex number here
                neighbors = self.getNeighborhood(candidate,layerNumber)
                for neighbor in neighbors:
                    # neighbor is a vertex number
                    # realNeighbor is the data tuple
                    realNeighbor = self.vertices[neighbor]
                    # checking if the neighbor is already in W or not
                    present = False
                    for element in W.arr:
                        if element[0] == neighbor:
                            present = True
                    if not present:
                        W.insert((neighbor, distanceFunction(e,realNeighbor)))
        # initializing minHeap for discarded connections
        # this is a minHeap because if keepPruned is true, we might need to reference the distance again
        WDiscard = minHeap()
        # putting M best candidates from W into R
        while W.size()>0 and R.size()<M:
            closest = W.extractMin()
            # comparing closest in W with largest in R, if applicable
            if R.size()==0 or R.peekMax()[1] > closest[1]:
                R.insert(closest)
            else:
                WDiscard.insert(closest)
        # if we have less than M elements in R, lets the closest discarded elements in R
        if keepPrunedConnections:
            while WDiscard.size()>0 and R.size()<M:
                closest = WDiscard.extractMin()
                R.insert(closest)

        # returning the max heap of up to M neighbors from the candidate list C
        return R
    # enter points are the entrance points of the layer (list of vertex numbers)
    # e is the query element
    # numNearest is the size of the dynamic candidate array
    # layerNumber is the layer of interest to search in the graph
    def searchLayer(self,e,enterPoints,numNearest,layerNumber, distanceFunction):
        v = deepcopy(enterPoints)
        distanceArray = deepcopy(v)
        # checking if we have 0 enterpoints (first insertion basically)
        if len(v) == 1 and v[0] is None:
            # returning an empty maxHeap
            return maxHeap()
        for i in range(len(distanceArray)):
            dataTuple = self.vertices[distanceArray[i]]
            distanceArray[i] = (distanceArray[i], distanceFunction(e,dataTuple))
        C = minHeap()
        C.heapify(deepcopy(distanceArray))
        W = maxHeap()
        W.heapify(distanceArray)
        while C.size()>0:
            closest = C.extractMin()
            #print(closest)
            furthest = W.peekMax()
            #print(furthest)
            if closest[1] > furthest[1]:
                # no more filtering to do
                break
            neighborhood = self.getNeighborhood(closest[0],layerNumber)
            for neighbor in neighborhood:
                present = False
                for element in v:
                    if element == neighbor:
                        present = True
                        break
                if not present:
                    neighborData = self.vertices[neighbor]
                    D = distanceFunction(e,neighborData)
                    v.append(neighbor)
                    # possibly updating C and W
                    if D < furthest[1] or W.size()<numNearest:
                       C.insert((neighbor,D))
                       W.insert((neighbor,D))
                       if W.size()>numNearest:
                           # removing the "worst" distance element from W
                           W.extractMax()
        # returning maxHeap of nearest neighbors of e in the given layer based on neighbors of enter points
        return W
    # main function for constructing the multi layer graph
    # newElement is the element to insert
    # M is the number of established connections
    # MMax is the max allowed connections on a vertex
    # constructParam is the size of the dynamic candidate list
    # normalizer is a normalizing constant for level choosing
    # usingHeuristic is a flag where if true, we heuristically select neighbors, where if false, we simple select
    def insertElement(self, newElement, M, MMax, MMax0, constructParam, normalizer, distanceFunction, usingHeuristic, keepPrunedConnections, extendCandidates=False):
        # firstly adding the newElement data to our list of vertices to keep a vertex number
        self.vertices.append(newElement)
        # expanding current layers with an extra space for the new element
        for layer in self.layers:
            layer.append(None)
        # index of vertex in the vertex list
        vertexNumber = len(self.vertices)-1
        enterPoint = [self.enterPoint]
        # getting top level (where the enter point is)
        topLevel = len(self.layers)-1
        # getting the new elements level based on decaying logarithm
        randomUniform = random()
        # will have an issue if random actually gives 0 because of the logarithm
        # if it ever gives precisely 0, we just map it to 1
        if randomUniform == 0:
            randomUniform = 1
        newLayer = int(-log(randomUniform) * normalizer)
        #print("adding element: " + str(newElement) + " to layer: " + str(newLayer))
        # if the new layer has not yet been created, we need to propogate our layers
        if newLayer > len(self.layers)-1:
            # then we need to extend our layers until this layer has been created
            while len(self.layers)-1 != newLayer:
                newEdgeList = []
                for i in range(len(self.vertices)):
                    newEdgeList.append(None)
                # appending the edge list to the layer
                self.layers.append(newEdgeList)
        # newLayer+2 since range is not inclusive
        for i in range(topLevel,newLayer+2,-1):
            #print("starting greedy part 1")
            # ef = 1
            W = self.searchLayer(newElement,enterPoint,1,i,distanceFunction)
            # above is a maxHeap
            # tuple of vertex number and distance from newElement
            smallestTuple = None
            for element in W.arr:
               if smallestTuple is None or smallestTuple[1]>element[1]:
                   smallestTuple = (element[0], element[1])
            # setting enter point for the future as the closest neighbor ("greedy")
            if smallestTuple is None:
                enterPoint = [None]
            else:
                enterPoint = [smallestTuple[0]]
        for i in range(min(topLevel,newLayer),-1,-1):
            #print("starting part 2")
            W = self.searchLayer(newElement,enterPoint,constructParam,i,distanceFunction)
            #print("done searching")
            # getting just array of vertex numbers
            vertexNumbers = []
            for element in W.arr:
                vertexNumbers.append(element[0])
            neighbors = None
            if usingHeuristic:
                neighbors = self.selectNeighborsHeuristic(distanceFunction,newElement,vertexNumbers,M,i,keepPrunedConnections,extendCandidates)
            else:
                neighbors = self.selectNeighborSimple(newElement,vertexNumbers,M,distanceFunction)
            # setting edges from newElement to neighbors in the graph at layer i
            layerAdjacencyList = self.layers[i]
            # neighbors are already in the list
            for neighbor in neighbors.arr:
                # adding to front of the list
                #print(neighbor)
                newNode = adjacencyNode(neighbor[0])
                anotherNode = adjacencyNode(vertexNumber)
                # adding to front of list for the neighbor
                anotherNode.next=layerAdjacencyList[neighbor[0]]
                layerAdjacencyList[neighbor[0]]=anotherNode
                # adding to front of list for the new element
                newNode.next = layerAdjacencyList[vertexNumber]
                layerAdjacencyList[vertexNumber] = newNode
            # shrinking connections if needed
            for neighbor in neighbors.arr:
                innerNeighborhood = self.getNeighborhood(neighbor[0],i)
                maxConnectionsAllowed = MMax
                if i ==0: maxConnectionsAllowed = MMax0
                if len(innerNeighborhood)>maxConnectionsAllowed:
                    # then we need to shrink the neighborhood
                    newNeighborhood = None
                    if usingHeuristic:
                        newNeighborhood = self.selectNeighborsHeuristic(distanceFunction, self.vertices[neighbor[0]], innerNeighborhood,maxConnectionsAllowed,i,keepPrunedConnections,extendCandidates)
                    else:
                        newNeighborhood = self.selectNeighborSimple(self.vertices[neighbor[0]],innerNeighborhood,maxConnectionsAllowed,distanceFunction)
                    # shrinking the neighborhood (this breaks bidirectionality, turns directed)
                    layerAdjacencyList = self.layers[i]
                    # removing the original edge list
                    layerAdjacencyList[neighbor[0]] = None
                    # adding the new edges, forming the new neighborhood
                    for newNeighbor in newNeighborhood.arr:
                        newNode = adjacencyNode(newNeighbor[0])
                        newNode.next = layerAdjacencyList[neighbor[0]]
                        layerAdjacencyList[neighbor[0]] = newNode
            # setting the new enter points
            enterPoint = vertexNumbers
        if newLayer > topLevel:
            self.enterPoint = vertexNumber
        # done with insertion
        #print("finished inserting element: " + str(newElement))
    # actual function for knn search
    # element is the element to use as basic for knn comparisons
    # k is the number of neighbors for knn
    # numNearest is the size of the candidate array during search
    def knnSearch(self, element,k, numNearest, distanceFunction):
        # grabbing the enter point
        enterPoints = [self.enterPoint]
        topLayerIndex = len(self.layers)-1
        for i in range(topLayerIndex,0,-1):
            W = self.searchLayer(element,enterPoints,1,i,distanceFunction)
            # result of search layer is a max heap, so we need linear time here to find the minimum in the array
            minNeighbor = None
            for neighbor in W.arr:
                if minNeighbor is None or minNeighbor[1] > neighbor[1]:
                    minNeighbor = neighbor
            # now minNeighbor is a tuple of (vertex, distance) representing nearest neighbor in W to element
            # setting the next enter point as the nearest neighbor in the current layer
            enterPoints = [minNeighbor[0]]
        # getting final neighbor list from layer 0 (the most dense layer)
        W = self.searchLayer(element,enterPoints,numNearest,0,distanceFunction)
        # heapifying W as a min heap instead of a max heap
        WMin = minHeap()
        WMin.heapify(W.arr)
        # extracting K minimums from W
        knn = []
        while len(knn) < k:
            knn.append(WMin.extractMin()[0])
        # knn now holds the vertex numbers of the nearest neighbors to the inputted element
        # to get the datapoints, we can just reference the vertex list
        return knn
    # holistic function to construct the hierarchal navigable small worlds graph from data points
    # reasonable value for M according to paper is from 6 to 48
    # setting MMax to M or 2M resulted in the best performance according to the paper
    # paper got good performance by setting efconstruction to 100
    # anything higher got slightly better quality at the cost of significantly increased construction time
    def buildGraph(self,distanceFunction, data, M, MMax, MMax0, efconstruction, usingHeuristic, keepPrunedConnections, extendCandidates=False):
        # paper provided the optimal normalizing constant: 1/ln(M)
        normalizingConstant = 1/log(M)
        # building the multi-layer graph, one datapoint at a time
        for datapoint in data:
            self.insertElement(datapoint,M,MMax,MMax0,efconstruction,normalizingConstant,distanceFunction,usingHeuristic,keepPrunedConnections,extendCandidates)





