from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import statistics 

import itertools
import cv2

class VLAD(object):
    def __init__(self):
        self.path = "src/data"
        self.descriptorName = "ORB"
        self.desc_path = "src/VPRdata/descriptorORB"
        self.pathVD = "src/VPRdata/visualDictionary2ORB" #output2
        self.vlad_desc = "src/VPRdata/VLAD_ORB_W2" #output3
        self.treeIndex = "src/VPRdata/index_ORB_W2" #output4
        self.queryResults = []
        

    def train(self):
        #computing the descriptors
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
        descriptors=getDescriptors(self.path, dict[self.descriptorName])
        print(descriptors.shape)
        #   writting the desc_path
        file= self.desc_path + ".pickle"

        with open(file, 'wb') as f:
            pickle.dump(descriptors, f)

    ######################################################################################################################################################

        desc_path = self.desc_path
        k = 100#256

        #computing the visual dictionary
        print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+desc_path)

        # with open(desc_path, 'rb') as f:
            # descriptors=pickle.load(f)

        visualDictionary=kMeansDictionary(descriptors,k)

        #self.desc_path
        file=self.pathVD + ".pickle"

        print("The visual dictionary  is saved in "+file)
        with open(file, 'wb') as f:
            pickle.dump(visualDictionary, f)

        #######################################################################################################################################################


        #estimating VLAD descriptors for the whole dataset


        print("estimating VLAD descriptors using "+self.descriptorName+ " for dataset: /"+self.path+ " and visual dictionary: /"+self.pathVD)
        with open(self.pathVD+ ".pickle", 'rb') as f:
            visualDictionary=pickle.load(f) 

        #computing the VLAD descriptors
        dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
        V, idImages = getVLADDescriptors(self.path,dict[self.descriptorName],visualDictionary)

        #output
        file= self.vlad_desc +".pickle"

        with open(file, 'wb') as f:
            pickle.dump([idImages, V,self.path], f)

        print("The VLAD descriptors are  saved in "+file)

    #######################################################################################################################################################

        leafSize = 40

        #estimating VLAD descriptors for the whole dataset
        print("indexing VLAD descriptors from "+self.vlad_desc+ " with a ball tree:")

        #load the vlad descriptors VD=[imageID, VLADdescriptors, pathToImageDataSet]
        with open(self.vlad_desc+ ".pickle", 'rb') as f:
            VLAD_DS=pickle.load(f)

        imageID=VLAD_DS[0]
        V=VLAD_DS[1]
        pathImageData=VLAD_DS[2]
        print(V)
        tree = indexBallTree(V,leafSize)
        #output
        file = self.treeIndex+".pickle"

        with open(file, 'wb') as f:
            pickle.dump([imageID,tree,pathImageData], f,pickle.HIGHEST_PROTOCOL)

        print("The ball tree index is saved at "+file)

#######################################################################################################################################################
#######################################################################################################################################################
#QUERY
#######################################################################################################################################################
#######################################################################################################################################################


    def query(self):
        k= 5
        pathVD = self.pathVD +".pickle"
        treeIndex = self.treeIndex +".pickle"


        def query_single_image(path, k, descriptorName, pathVD, treeIndex):
            #load the index
            with open(treeIndex, 'rb') as f:
                indexStructure=pickle.load(f)

            #load the visual dictionary
            with open(pathVD, 'rb') as f:
                visualDictionary=pickle.load(f)     

            imageID=indexStructure[0]
            tree = indexStructure[1]
            pathImageData = indexStructure[2]

            print(pathImageData)
            #computing descriptors
            dist,ind = query(path, k,descriptorName, visualDictionary,tree)

            print(dist)
            print(ind)
            ind=list(itertools.chain.from_iterable(ind))

            print(path)
            # display the query
            imageQuery=cv2.imread(path)
            #cv2.imshow("Query", imageQuery)
            #cv2.waitKey(0);

            # loop over the results
            print("Before for: ", imageID[1])
            
            for i in ind:
                # load the result image and display it
                result = cv2.imread(imageID[i])
                list_ = imageID[i].split('_')[0]
                list_ = int(list_.split('/')[2])
                self.queryResults.append(list_)
                print("list: ", list_)
                cv2.imshow("Result", result)
                #cv2.waitKey(0)

            print("queryResults: ", self.queryResults)


        paths = ['src/queries/0_img.png', 'src/queries/1_img.png', 'src/queries/2_img.png', 'src/queries/3_img.png' ]
        for path in paths:
            query_single_image(path, k, self.descriptorName, pathVD, treeIndex)
        return statistics.median(self.queryResults)