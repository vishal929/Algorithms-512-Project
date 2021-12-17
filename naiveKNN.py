# idea here is to just have code for naive k-nn

from dataStructures import maxHeap



# naive knn with array of neighbors
# e is the element to get neighbors for
# data is the array of data points to compare e to
# k is the number of nearest neighbors to grab
def naiveKNNArray(e, data, k, distanceFunction):
    nearest = []
    for candidate in data:
        # computing distance based on the given distance function
        distance = distanceFunction(e,candidate)
        # checking if we can add the data point to the nearest array or not
        if len(nearest) < k:
            # keeping the distance in the tuple, so we do not have to recompute it
            nearest.append((candidate,distance))
        else:
            # checking if the computed distance above is < the maximum distance of a current neighbor
            # if so, then we can swap them
            maxDistance = None
            maxDistanceIndex = None
            # computing the maximum
            for i in range(len(nearest)):
                if maxDistance is None:
                    maxDistanceIndex = i
                    maxDistance = nearest[i][1]
                elif nearest[i][1] > maxDistance:
                    maxDistance = nearest[i][1]
                    maxDistanceIndex = i
            if maxDistance > distance:
                # then we can swap the candidate and the maximum distance neighbor in the array
                nearest[maxDistanceIndex] = (candidate,distance)
    # returning the nearest neighbors to e in the dataset
    return nearest

# same as above, but with a max heap of fixed size k to help with comparison
# returns a maxHeap, so we can get an array from it if we wish afterwards
def naiveKNNHeap(e, data, k, distanceFunction):
    # having nearest be a maxHeap instead of an array (for faster maximum distance comparison)
    nearest = maxHeap()
    for candidate in data:
        # computing distance based on the given distance function
        distance = distanceFunction(e, candidate)
        # checking if we can add the data point to the nearest array or not
        if nearest.size()<k:
            # inserting value, key pair into the max heap
            nearest.insert((candidate,distance))
        else:
            # checking if the computed distance above is < the maximum distance of a current neighbor
            # if so, then we can swap them
            maxNeighborTuple = nearest.peekMax()
            if maxNeighborTuple[1] > distance:
                # then we delete this neighbor and insert the candidate instead
                nearest.extractMax()
                nearest.insert((candidate,distance))
    # returning the nearest neighbors to e in the dataset as a maxHeap object
    return nearest
