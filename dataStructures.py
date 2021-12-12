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
    # function to sift up
    def siftUp(self):
        # fill out
    # function to sift down
    def siftDown(self):
        # fill out
    # function to insert into the queue
    # e is the element to insert
    # e should be formatted as the following tuple: (data, distance)
    def insert(self,e):
        # need to insert the element as a leaf and sift up
        self.arr.append(e)
        # sifting up
    # function to extract a maximum element from the maxHeap
    def extractMax(self):
        # return the root, and then replace the root value with the last leaf, and then sift down


# class for kd trees datastructure
class kdTree(object):
    def __init__(self):
        # do stuff
    def constructTree(self,data):
        # constructing tree from dataset

# class for the multilayer graph to be implemented as part of the hierarchal small world graphs implementation
class multiLayerGraph(object):
    def __init__(self):
        # do stuff