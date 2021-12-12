# python file that houses our defined data structures and some helper functions

# helper function to get the median of the dataset based on a specific feature
def medianFeature(dataArray,featureToUse):


# helper function to pivot a dataset based on a certain feature value
# pivots the array about the median of the array
def partitionFeature(dataset, i,j, featureToPivot, indexToPivot):
    lower = []
    higher = []

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
        self.arr[0] = self.arr[len(self.arr)-1]
        self.arr.pop()
        self.siftDown(0)

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
        mid = pivotFeature(self.dataset,i,j,featureToSplit)
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



# class for the multilayer graph to be implemented as part of the hierarchal small world graphs implementation
class multiLayerGraph(object):
    def __init__(self):
        # do stuff
        stuff =1