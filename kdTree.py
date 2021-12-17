# houses logic for knn with kd trees

import dataStructures as ds
import dataGrabber

# lets test it on the iris dataset

# grabbing the iris dataset
irisDataset = dataGrabber.grabIris()

# declaring the kd tree
kdTree = ds.kdTree(irisDataset,4)

#printing all the nodes in the tree
kdTree.printTree(kdTree.rootNode)

# kd tree is automatically constructed from the data

# lets test some knn
nearest = kdTree.kdTreeKNN(irisDataset[0],5,dataGrabber.irisDistanceFunction)
print(nearest.arr)


