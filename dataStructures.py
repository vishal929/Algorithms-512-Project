# python file that houses our defined data structures and some helper functions


# helper function for selection
# selects kth smallest
def select(dataset,i,j,featureForSelection, k):
    pass

# helper function to run insertion sort on part of an array based on a specific feature
def insertionSortFeature(dataset, i,j, featureToUse):
    # sorts the array based on the feature using insertion sort
    for k in range(i+1,j+1):
        value = dataset[k][featureToUse]
        lastIndex = k-1
        lastValue = dataset[lastIndex][featureToUse]
        while value < lastValue:
            lastIndex -= 1
            lastValue = dataset[lastIndex][featureToUse]
        # swap lastIndex +1 with the current k value
        temp = dataset[lastIndex+1]
        dataset[lastIndex+1] = dataset[k]
        dataset[k] = temp
    # array is sorted in place



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
            parent = (index-1)/2
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
            return
        # swapping root with leaf
        self.arr[0] = self.arr[len(self.arr)-1]
        # removing the leaf
        self.arr.remove(len(self.arr)-1)
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
            parent = (index-1)/2
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
            return
        # swapping root value with a leaf value
        self.arr[0] = self.arr[len(self.arr)-1]
        # removing the leaf value
        self.arr.remove(len(self.arr)-1)
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
    def __init__(self,data):
        self.dataset = data
        # getting the number of features in the dataset based on the length of a data tuple
        self.numFeatures = len(data[0])
        # constructing the kdtree from data and setting the root node
        self.rootNode = self.constructTreeDriver()
    # recursive construction of a kdtree from a dataset
    # data is an array of element tuples where parts of the tuple are features
    # i and j are indices for recursion
    # featureToSplit is just the feature number to split the next level on (dimension)
    def constructTree(self,i,j,featureToSplit):
        # recursively constructing tree from the dataset for the kdtree
        # resetting the feature split to "wrap around"
        if featureToSplit > self.numFeatures:
            featureToSplit=0

        if i == j:
            # then we only have 1 piece of data in this subtree
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
        # incrementing the feature to split on next
        featureToSplit += 1
        parent = kdNode(self.dataset[mid],featureToSplit, None, None)
        parent.left = self.constructTree(i,mid-1,featureToSplit)
        parent.right = self.constructTree(mid+1,j,featureToSplit)

        # returning the parent
        return parent

    # driver to construct a tree recursively and set it as the root node
    # this is called during init with a dataset of tuples
    def constructTreeDriver(self):
        self.rootNode = self.constructTree(0,len(self.dataset)-1,0)

class adjacencyVertices(object):
    def __init__(self,data):
       self.data = data

class adjacencyNode(object):
    def __init__(self, vertex):
        # vertex associated with the adjacency edge (vertex number)
        self.vertex = vertex
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
        self.vertices = []
        # self.layers is a list of layers of adjacency lists for the hierarchal structure
        # adjacency lists have a vertex number
        self.layers = []
        # double list because each layer contains a list of linked lists (each layer has an adjacency list)
    # gets neighborhood of a node in a given layer
    # v is the vertex to get the neighborhood of (v has an index)
    # G is the multi layer graph
    # layerNumber is the layer to consider in neighbor-grabbing
    def getNeighborhood(self, v, layerNumber):
        neighbors = []
        startNode = self.layers[layerNumber][v]
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
    def selectNeighborSimple(self, e, C, M, distanceFunction):
        # initializing a max heap of size M
        Q = maxHeap()
        for candidate in C:
            # calculating the distance
            D = distanceFunction(e, candidate)
            if Q.size() < M:
                Q.insert((candidate, D))
            else:
                if Q.peekMax()[1] > D:
                    Q.extractMax()
                    Q.insert((candidate, D))
        # returning the queue of up to M neighbors from the candidate list C
        return Q

    # heuristic method of getting neighbors
    def selectNeighborsHeuristic(self, e, C, M, layerNumber, keepPrunedConnections, extendCandidates=False):
        # R is a max heap
        R = maxHeap()
        # min heap of candidate datapoints
        # assuming that C has input as an array, so we have to heapify into W
        W = minHeap().heapify(C)
        if extendCandidates:
            for candidate in C:
                neighbors = self.getNeighborhood()