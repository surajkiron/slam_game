python describe.py --dataset data --descriptor ORB --output descriptors/descriptorORB
python visualDictionary.py -d descriptors/descriptorORB.pickle  -w 256 -o visualDictionary/visualDictionary2ORB
python vladDescriptors.py  -d data -dV visualDictionary/visualDictionary2ORB.pickle --descriptor ORB -o VLADdescriptors/VLAD_ORB_W2
python indexBallTree.py  -d VLADdescriptors/VLAD_ORB_W2.pickle -l 40 -o ballTreeIndexes/index_ORB_W2

