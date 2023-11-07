from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import statistics 

import itertools
import cv2

class VLAD(object):
    def __init__(self):
        self.descriptorName = "ORB"
        self.queryResults = []
        self.indexStructure = []
        self.visualDictionary = None
        self.min_variance = 1e6

    def train(self, train_imgs):
        #computing the descriptors
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
        descriptors=getDescriptors(train_imgs, dict[self.descriptorName])


    ######################################################################################################################################################


        #computing the visual dictionary


        k = 16 # Size of Visual Descriptors 
     
        self.visualDictionary=kMeansDictionary(descriptors,k)


        #computing the VLAD descriptors


        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
        V, imageID = getVLADDescriptors(train_imgs,dict[self.descriptorName],self.visualDictionary)


        #estimating VLAD descriptors for the whole dataset

        leafSize = 40
        tree = indexBallTree(V,leafSize)
        self.indexStructure = [imageID,tree]
        
        return(0)

#######################################################################################################################################################
#######################################################################################################################################################
#QUERY
#######################################################################################################################################################
#######################################################################################################################################################


    def query(self):

        def query_single_image(path, k, descriptorName):
            
            tree = self.indexStructure[1]

            #computing descriptors
            _,ind = query(path, k,descriptorName, self.visualDictionary,tree)

            ind=list(itertools.chain.from_iterable(ind))
            variance = statistics.variance(ind)
            if variance < self.min_variance: 
                self.min_variance = variance
                self.queryResults.append(statistics.median(ind))



        k= 5

        paths = ['src/queries/0_img.png', 'src/queries/1_img.png', 'src/queries/2_img.png', 'src/queries/3_img.png' ]
        for path in paths:
            query_single_image(path, k, self.descriptorName)
        print("queryResults: ", self.queryResults)
        return self.queryResults[::-1]