# Custom-DeTr


## Step-1)
We were having labeled dataset for panoptic segmentation but only things were labeled in the images not stuff. so, we used pretrained DeTr on coco dataset for getting the penoptic segmentation for all the classes. 

## Step-2)
Finally we had two things first is ground truth which was labeled by us and prediction of DeTr weights. we mapped both the outputs together for getting labels for all the classes 

## Step-3)
One Problem was the predifined coco classes. so, we treated them as misslanious stuff for the model. These classes includes Person, Car, Airoplane etc..

## Step-4)
Finally we had the labeled data so, splited it in train and test set for training in ration of 70:30, Cloned the Github Repository of DeTr and trained the model on custom classes
