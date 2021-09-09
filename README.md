# Custom-DeTr

# The Problem Statement

DATASET CAN BE DOWNLOADED FROM THIS LINK (Links to an external site.)
Train DETR on this dataset!
Explain the additional questions above
Submit all the annotated images by 21st August. Dataset will be shared on 19th August.
The project is divided into 3 parts with 3 sequential submissions:
Submit (on the 7th day from today)  the questions asked above with a readme file explaining how are you going to solve this problem. 
Submit (on the 20th day from today) the model trained for BBOX detection on your dataset (combined with stuff)
Submit (on the 30th day from today) the final model trained for segmentation on your model
You need to document your training process, and link separate (properly named) notebooks in the readme file along with notes for me to understand how you have trained your model. Your training logs MUST be available for review.
This is not a group assignment, and your code must not match any other student.
You need to document your training process, and link separate (properly named) notebooks in the readme file along with notes for me to understand how you have trained your model. Your training logs MUST be available for review.  
Linking the model end to end is not the objective. The objective is to solve the problem end to end. You MUST have trained for over 500 EPOCHS in total (for all the models combined) and show that your loss is reduced substantially from the starting point.
You need to split your dataset into 80/20 and then show the test accuracy in the readme file.
Missing the first 2 submissions by the 20th day will disqualify you from the CAPSTONE project.
You must show the results on 100 randomly picked samples from the test dataset and show the results in the following 3 parallel images:
First Image: Orignal Image
Second: Ground Truth
Third: Predicted Segmentation Map
I am going to read your README to understand the whole thing. If it doesn't cover something (like what loss function you used, or how the logs looked like) I would assume that it's not done and the assignment would be graded accordingly. 


## Step-1)
We were having labeled dataset for panoptic segmentation but only things were labeled in the images not stuff. so, we used pretrained DeTr on coco dataset for getting the penoptic segmentation for all the classes. 

## Step-2)
Finally we had two things first is ground truth which was labeled by us and prediction of DeTr weights. we mapped both the outputs together for getting labels for all the classes 

## Step-3)
One Problem was the predifined coco classes. so, we treated them as misslanious stuff for the model. These classes includes Person, Car, Airoplane etc..

## Step-4)
Finally we had the labeled data so, splited it in train and test set for training in ration of 70:30, Cloned the Github Repository of DeTr and trained the model on custom classes
