from VLADlib.VLAD import *
from VLADlib.Descriptors import *

path = "src/data"
descriptorName = "ORB"
output = "src/descriptors/descriptorORB"


#computing the descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
descriptors=getDescriptors(path, dict[descriptorName])
print(descriptors.shape)
#writting the output
file=output+".pickle"

with open(file, 'wb') as f:
	pickle.dump(descriptors, f)

######################################################################################################################################################

desc_path = output
k = 256
output2 = "src/visualDictionary/visualDictionary2ORB"

#computing the visual dictionary
print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+desc_path)

# with open(desc_path, 'rb') as f:
    # descriptors=pickle.load(f)

visualDictionary=kMeansDictionary(descriptors,k)

#output
file=output2 + ".pickle"

with open(file, 'wb') as f:
	pickle.dump(visualDictionary, f)

print("The visual dictionary  is saved in "+file)

#######################################################################################################################################################

pathVD = output2
output3="src/VLADdescriptors/VLAD_ORB_W2"



#estimating VLAD descriptors for the whole dataset
print("estimating VLAD descriptors using "+descriptorName+ " for dataset: /"+path+ " and visual dictionary: /"+pathVD)


with open(pathVD, 'rb') as f:
    visualDictionary=pickle.load(f) 

#computing the VLAD descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
V, idImages = getVLADDescriptors(path,dict[descriptorName],visualDictionary)

#output
file=output3+".pickle"

with open(file, 'wb') as f:
	pickle.dump([idImages, V,path], f)

print("The VLAD descriptors are  saved in "+file)

#######################################################################################################################################################

leafSize = 40
output4= "src/ballTreeIndexes/index_ORB_W2"

#estimating VLAD descriptors for the whole dataset
print("indexing VLAD descriptors from "+output3+ " with a ball tree:")

#load the vlad descriptors VD=[imageID, VLADdescriptors, pathToImageDataSet]
with open(output3, 'rb') as f:
    VLAD_DS=pickle.load(f)

imageID=VLAD_DS[0]
V=VLAD_DS[1]
pathImageData=VLAD_DS[2]
print(V)
tree = indexBallTree(V,leafSize)
#output
file=output4+".pickle"

with open(file, 'wb') as f:
	pickle.dump([imageID,tree,pathImageData], f,pickle.HIGHEST_PROTOCOL)

print("The ball tree index is saved at "+file)

#######################################################################################################################################################
#######################################################################################################################################################
#QUERY
#######################################################################################################################################################
#######################################################################################################################################################

import itertools
import cv2

k= 5
pathVD = output2
treeIndex = output4

paths = ['queries/0_img.png', 'queries/1_img.png', 'queries/2_img.png', 'queries/3_img.png' ]
for path in paths:
    query(path, k, descriptorName, pathVD, treeIndex)

def query(path, k, descriptorName, pathVD, treeIndex):
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
    for i in ind:
        # load the result image and display it
        print(imageID[i])
        result = cv2.imread(imageID[i])
        cv2.imshow("Result", result)
        #cv2.waitKey(0)



