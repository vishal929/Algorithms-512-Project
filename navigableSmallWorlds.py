# file gives an example of running the approximate KNN from hierarchical navigable small worlds

# algorithms provided by TA from:
# https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf

import dataStructures as ds
import dataGrabber

# putting knn logic here for small worlds

# application to iris dataset

# grabbing iris dataset
irisData = dataGrabber.grabIris()

# declaring a small worlds object
smallWorldsGraph = ds.multiLayerGraph()

# building the graph based on the distance function
# paper suggests M between 5 and 48, lets pick M=10
# lets set MMax = 2M
# constructionParam should be 100
M = 10
smallWorldsGraph.buildGraph(dataGrabber.irisDistanceFunction,irisData,M,M,2*M,100,True,True)

# testing knn
res = smallWorldsGraph.knnSearch(irisData[0],5,30,dataGrabber.irisDistanceFunction)
actualRes = []
for vertexNum in res:
    actualRes.append(smallWorldsGraph.vertices[vertexNum])

# printing results
print(actualRes)
print("finished")

