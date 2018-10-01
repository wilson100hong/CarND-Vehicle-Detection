---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: images/vehicle_hog.png
[image2]: images/nonvehicle_hog.png
[image3]: images/scale.png
[image4]: images/classifier.png
[image5]: images/filter_1.png
[image6]: images/filter_2.png
[image7]: images/label.png
[Video]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

Writeup: README.md

Video: [Video]   [YouTube](https://youtu.be/hycKd8b34gY)   

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
Before that, I correct the input image with a camera distortion corrector `camera.py`.

I use `kimage.feature.hog()` to extract HOG features from converted YUV images among all(three) channels.
The code for this step is in `utils.py` line. 88 - 140.

Examples for HOG images in `vehicle` and `non-vehicle` directories:

* Vehicle
![vehicle hog][image1]

* Non-vehicle
![non-vehicle hog][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.
I did experiments on different combinations of parameters. I evaluate the result based on:
* Testing accuracy in training.
* Vehicle detection sensitivity.
* Computation efficiency.

My observations:
* For `orientations`, any number great than `9` has similar training accuracy (all 99.3+%). So I choose the smallest one.
* For `pexles_per_cell`, it is ideal to choose a number which can divide training image, which is 64x64. I also prefer to use
  power of 2 because of simplicity. The candidates are `8` and `16`. I finally choose `8`, because it is more sensitive in
  vehicle detection in sliding window. 
* For `cells_per_block`, I tried `2` and `4`. `2` gives better training accuracy so I use `2`.
* For `block_norm`, I just used what opencv suggested.
* For `transform_sqrt`, I use `False` based on opencv suggestion of avoid global ambiance.

The final parameters I used:

| Parameter       | Value    |
|-----------------|----------|
| orientations    | 9        |
| pixels_per_cell | 8        |
| cells_per_block | 2        |
| block_norm      | 'L2-Hys' |
| transform_sqrt  | False    |



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is in `vehicle.py` the `Trainer` class, line. 42 - 138.
1. Loads images from `vehicle` and `non-vehicle` directories
2. Extracts the following three features. Extracted features are flatten into 1-d vector.
  * Spatial bin

    | Parameter       | Value    |
    |-----------------|----------|
    | spatial size    | 16 x 16  |
    
  * Color histogram bin
    
    | Parameter       | Value    |
    |-----------------|----------|
    | window size     | 64 x 64  |
    | color bins      | 16       |
        
  * HOG

3. Split dataset into training set and testing set (8:2) by using `sklearn.cross_validation.train_test_split()`
4. Fit per-column scaler on training set by using `StandardScaler`.
5. Shuffles the training set.
6. Train classifier by using linear SVM `svm.LinearSVC()`. Testing accuracy: 99.3%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code is in `vehicle.py` function `search_cars_with_option()`, line. 155 - 230.
It is based on the global-HOG sub-sampling method introduced in the class.
1. Extract global HOG
2. Scales the source image.
3. Slide the window based on `cells_per_step`, for each window , extract all features and pass to classifier.
4. If classifier has high confidence, adds the window with the confidence score.

Since vehicles in video have different sizes then training data (64x64), I decided to use multiple scales 
to better compensate with each others. I tried different scales: (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
0.75 and 2.5+ cannot detect anything so I throw them away.
1.25 behaves pretty much the same as 1.0, and 1.76 is the same as 1.5. I discard them because of redundancy.

 
For overlapping, I first tried 87.5% overlapping (`cells_per_step`=1), which
gives a pretty good result. However, 87.5% overlapping is too sensitive and cause a lot of false positive detections, and 
it is also very slow (around 5 seconds in each frame).

I then tried 75% overlapping (`cells_per_step`=2), which also gives acceptable result in detection. But it is not that sensitive
compared to 87.5% overlapping, especially at the edge of car or occlusions. I finally decided to use 75% because of efficiency (2.5 seconds per frame).

Scales and overlapping and example image:

| Color | Scale | Overlapping | cells per block |
|-------|-------|-------------|-----------------|
| red   | 1.0   | 75%         | 2               |
| blue  | 1.5   | 75%         | 2               |
| green | 2.0   | 75%         | 2               |

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Convert to YUV image, use spatial bins, color histogram and 3-channesl HOG. 
Using 3 sets of scales and overlapping.
Additionally, I did "camera correction" based on the distortion coefficients computed in project 4.

Examples from test images:
![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Video: [YouTube](https://youtu.be/hycKd8b34gY)

[Video] 

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
I did several things to filter false positives:
1. Filter out low confidence detection with a "confidence threshold", see `vehicle.py` line. 222.
2. For window detected by classifier (a bounding boxes with confidence scores), I create a heatmap and put them on the map.
   Each window contribute its confidence to its contained area. This accumulated confidence map is called "naive heatmap", see `vehicle.py` line. 303.
3. Filter out detected windows which the maximum confidence score of its inner pixel is less than "bbox confidence threshold".
   This guarantees the remaining windows contain a peak pixel with high confidence. See `vehicle.py` line. 257.
4. Create another heatmap and put the remaining windows on it.
5. Filter that heatmap with "heatmap threshold". This helps the heatmap bounds vehicles' edges tighter.
6. Add the heatmap with exponential decay factor 0.8 to history heatmap. This helps to smooth the detection with a tiny delay in detection.
7. Use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
8. Use the blobs as the vehicles.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Detected windows and naive heatmaps from test images

![alt text][image5]

### Filtered windows and thresholded heatmaps from test images

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` labels:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. The biggest problem I faced in this project is false positive detection, especially when signs and shadows no landmarks.
   - Using the classifier's confidence score instead of using the prediction result directly helps reduce such cases in pipeline beginning.
   - Using "bbox confidence threshold", I successfully removed more FP cases cause by sparse, low confidence window overlapping.
   - Furthermore, use exponential decay helps to preserve temporal detection results and smooth the detection.
   The result is pretty satisfying, with tiny additonal delay.
   
2. Wobbly bounding box
   The bounding box of vehicles is wobbing alot, I think the problem is due to the exponential decay part does not preserve 
   historical result that long and strong. 0.8 decay factor might be too large (detection fade out in 10 frames). A momery window 
   or lower decay factor might help.
   
3. Overlapping / segmentation
   The pipeline cannot segmented cars when they overlapped with each other. This is because `label()` counts vehicles as one blob when they connect in heatmap.
   We might be able to use a algorithm which can cluster heatmap into multiple clusters givens the number of "peaks in heatmap".
