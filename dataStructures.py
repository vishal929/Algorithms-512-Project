# python file that houses our defined data structures

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
        siftStack = []
        siftStack.append(index)
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
        siftStack = []
        siftStack.append(index)
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


# class for kd trees datastructure
class kdTree(object):
    def __init__(self):
        # do stuff
        stuff =1
    def constructTree(self,data):
        # constructing tree from dataset
        stuff =1

# class for the multilayer graph to be implemented as part of the hierarchal small world graphs implementation
class multiLayerGraph(object):
    def __init__(self):
        # do stuff
        stuff =1