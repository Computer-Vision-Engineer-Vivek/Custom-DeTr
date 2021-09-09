# Object Detection Phase-2

# Custom-DeTr

# The Problem Statement

- DATASET CAN BE DOWNLOADED FROM THIS LINK (Links to an external site.)
- Train DETR on this dataset!
- Explain the additional questions above
- Submit all the annotated images by 21st August. Dataset will be shared on 19th August.
- Submit (on the 20th day from today) the model trained for BBOX detection on your dataset (combined with stuff)
- You need to document your training process, and link separate (properly named) notebooks in the readme file along with notes for me to understand how you have trained your model. Your training logs MUST be available for review.
- This is not a group assignment, and your code must not match any other student.
- You need to document your training process, and link separate (properly named) notebooks in the readme file along with notes for me to understand how you have trained your model. Your training logs MUST be available for review.  
- Linking the model end to end is not the objective. The objective is to solve the problem end to end. You MUST have trained for over 500 EPOCHS in total (for all the models combined) and show that your loss is reduced substantially from the starting point.
- You need to split your dataset into 80/20 and then show the test accuracy in the readme file.
- Missing the first 2 submissions by the 20th day will disqualify you from the CAPSTONE project.
- You must show the results on 100 randomly picked samples from the test dataset and show the results in the following 3 parallel images:
- First Image: Original Image
- Second: Ground Truth
- I am going to read your README to understand the whole thing. If it doesn't cover something (like what loss function you used, or how the logs looked like) I would assume that it's not done and the assignment would be graded accordingly. 


## Step-1) How was the labeling handled?
We were having a labeled dataset for panoptic segmentation but only things were labeled in the images not stuff. so, we used pretrained DeTr on coco dataset for getting the penoptic segmentation for all the classes. 
In this procedure the aim was to get the predicted mask by the pretrained weights for getting the stuff classes. 

## Step-2) What about the two separated outputs of DeTr Prediction & Ground Truth?
Finally we had two things: the first is ground truth which was labeled by us and prediction of DeTr weights. we mapped both the outputs together for getting labels for all the classes 
In this case we also faced the issue of overlapping coordinates for the classes we labeled and the same classes predicted by the DeTr.

## Step-3) How the Overlapping Problem was handled & what about the predefined coco classes which was predicted by the DeTr?
One Problem was the predefined coco classes. so, we treated them as miscellaneous stuff for the model. These classes include Person, Car, Airplane etc..
Also the overlapping area was considered  as miscellaneous stuff for all the images

## Step-4) Final step
Finally we had the labeled data so, splitted it in train and test set for training in ration of 80:20, Cloned the Github Repository of DeTr and trained the model on custom classes
