# houses logic for knn with kd trees

import dataStructures as ds
import dataGrabber
import naiveKNN

# lets test it on the iris dataset

# grabbing the iris dataset
irisDataset = dataGrabber.grabIris()

# testing insertion sort feature
'''
print("presorted: " + str(irisDataset[0:6]))
ds.insertionSortFeature(irisDataset,0,5,0)
print("postsorted: " + str(irisDataset[0:6]))
'''

# declaring the kd tree
kdTree = ds.kdTree(irisDataset,4)

#printing all the nodes in the tree
#kdTree.printTree(kdTree.rootNode)

# kd tree is automatically constructed from the data

# lets test some knn
nearest = kdTree.kdTreeKNN(irisDataset[0],5,dataGrabber.irisDistanceFunction)
print(nearest.arr)

# comparison of naive knn to kdtree knn (should be the same result)
near = naiveKNN.naiveKNNHeap(irisDataset[0],irisDataset,5,dataGrabber.irisDistanceFunction)
print(near.arr)


