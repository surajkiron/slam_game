from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import statistics 

import itertools
import cv2

# VLAD object for Vector of Locally Aggregated Descriptors
class VLAD(object):
    def __init__(self):
        self.descriptorName = "ORB"  # Default descriptor
        self.queryResults = []  # Stores results of queries
        self.indexStructure = []  # Indexing structure for the descriptors
        self.visualDictionary = None  # Visual dictionary for the VLAD descriptors
        self.min_variance = 1e6  # Minimum variance threshold

    # Method to train the VLAD model with given training images
    def train(self, train_imgs):
        # Computing the descriptors using the descriptor function mapped by descriptorName
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
        descriptors = getDescriptors(train_imgs, dict[self.descriptorName])

        # Computing the visual dictionary using K-means clustering
        k = 16  # Size of the visual dictionary (number of clusters)
        self.visualDictionary = kMeansDictionary(descriptors, k)

        # Computing the VLAD descriptors with the visual dictionary
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
        V, imageID = getVLADDescriptors(train_imgs, dict[self.descriptorName], self.visualDictionary)

        # Indexing the VLAD descriptors using a Ball Tree for fast nearest neighbor search
        leafSize = 40  # Size of the leaf in the Ball Tree
        tree = indexBallTree(V, leafSize)
        self.indexStructure = [imageID, tree]
        
        return(0)

    # Method to query the VLAD model with a set of images
    def query(self):
        # Nested method to query a single image
        def query_single_image(path, k, descriptorName):
            tree = self.indexStructure[1]  # Retrieve the Ball Tree from the index structure

            # Computing descriptors and finding the k nearest neighbors in the Ball Tree
            _, ind = query(path, k, descriptorName, self.visualDictionary, tree)

            # Flatten the list of indices and compute the variance
            ind = list(itertools.chain.from_iterable(ind))
            variance = statistics.variance(ind)
            
            # Update the minimum variance and append the median of the indices to the results
            if variance < self.min_variance: 
                self.min_variance = variance
                self.queryResults.append(statistics.median(ind))

        # Number of nearest neighbors to retrieve
        k = 5

        # Paths to the query images
        paths = ['/home/harshit/vis_nav_player/finGame/src/queries/0_img.png','/home/harshit/vis_nav_player/finGame/src/queries/1_img.png','/home/harshit/vis_nav_player/finGame/src/queries/2_img.png','/home/harshit/vis_nav_player/finGame/src/queries/3_img.png']

        # Query for each image path
        for path in paths:
            query_single_image(path, k, self.descriptorName)
        
        # Print the query results
        print("queryResults: ", self.queryResults)
        
        # Return the query results in reverse order
        return self.queryResults[::-1]
