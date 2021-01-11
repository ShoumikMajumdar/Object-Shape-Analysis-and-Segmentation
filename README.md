# Object-Shape-Analysis-and-Segmentation

<html>


<body>

<div class="main-body">
  <hr>
  <h2> Binary Image Analysis </h2>

  <h3> Problem Definition </h3>
  <p>
  Given a binary image (e.g. hands or tumor images), I try to find connected
  components and label each object. We try to detect boundary and skeleton of
  an object. I also calculate object area, orientation, circularity, and
  compactness for each object.
  </p>

  <h3> Method and Implementation </h3>
  <ol>
    <li>
    <p>
    Connected Component Labeling:  Scan every pixel in the image, find a pixel
    which is not background. After finding the pixel, we label the pixel as 1 at
    first. Then, I push the pixel in the stack and find neighbors whether they
    have same intensity value. If so, the neighbor also is pushed into stack and
    assigned the same label. Pop an item from the stack and keep search for
    neighbor’s of neighbors. After stack is empty, we increase the label count
    and find another pixel that needs to be labeled.
    </p>
    </li>
    <li>
    <p>
    I first added a padding of size 1px to the original image in order to be
    able to progress with the boundary following algorithm discussed in class.
    I implemented the boundary following algorithm by finding the first black
    pixel and start following the boundary.
    </p>
    </li>
    <li>
    <p>
    After I labeled all the objects from sequential connected component
    labeling, I passed the set of objects to compute its area by counting the
    number of pixels, orientation by computing Emin and Emax.
    </p>
    <p>
    I then compute the circularity by Emin/Emax. For compactness, I run the
    boundary following algorithm to get the perimeter for any object. Then the
    compactness is computed.
    </p>
    </li>
    <li>
    <p>
    Scan every pixel in the image, find a pixel which is not background. After
    finding the pixel, I compute the closest distance from any background
    pixels. Then, we compare this distance with distances from it neighbor’s to
    background. If it is bigger than that of neighbor’s, I classify it as a
    skeleton pixels.
    </p>
    </li>
  </ol>
  <h3> Experiments and Results </h3>

  <p>
  I tested our implementation on the four sample images.The results are in the table below.
  </p>
  <table>

    <tr>
      <td>Examples</td><td> Source </td><td> Labeling </td><td> Boundary Following </td> <td> Skeleton </td>
    </tr>
    <tr>
      <td> Example 1</td>
      <td> <img src="images/hand1.png" width="150" height="150"></td>
      <td> <img src="images/hand1_flood.jpg" width="150" height="150"></td>
      <td> <img src="images/hand1_boundary.jpg" width="150" height="150"></td>
      <td> <img src="images/hand1_skeleton.jpg" width="150" height="150"></td>

    </tr>
    <tr>
      <td> Example 2</td>
      <td> <img src="images/2.png" width="150" height="150"> </img></td>
      <td> <img src="images/hand2_fill.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/hand2_boundary.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/hand2_skel.jpg" width="150" height="150"> </img></td>
    </tr>
    <tr>
      <td> Example 3 </td>
      <td> <img src="images/hand3.png" width="150" height="150"> </img></td>
      <td> <img src="images/Hand3_flood.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/hand3_boundary.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/hand3_skeleton.jpg" width="150" height="150"> </img></td>

    </tr>
    <tr>
      <td> Example 4</td>
      <td> <img src="images/tissue.png" width="150" height="150"> </img></td>
      <td> <img src="images/tissue_filled.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/tissue_boundary.jpg" width="150" height="150"> </img></td>
      <td> <img src="images/tissue_skeleton.jpg" width="150" height="150"> </img></td>
    </tr>
  </table>
  <p>
  For are orientation an circularity,I got the following results:
  <table>
  <tr><td>Image</td><td>Results</td></tr>
  <tr><td><img src="images/hand1.png" width="150" height="150"></td><td>Perimeter = 1316.53108 Area = 30095.5 Compactness = 57.5918 Orientation = -40.1668 Circularity = 0.814714</td></tr>
  <tr><td><img src="images/2.png" width="150" height="150"></td><td>Perimeter = 280.6101 Area = 2568.5 Compactness = 30.6568 Orientation = 33.566 Circularity = 0.47468</td></tr>
  <tr><td><img src="images/hand3.png" width="150" height="150"></td><td>Perimeter = 683.22 Area = 23390 Compactness = 19.95 Orientation = 8.195 Circularity = 0.274 <br> Perimeter = 1190.35 Area = 30371 Compactness = 46.65 Orientation = -2.513 Circularity = 0.25</td></tr>
  <tr><td><img src="images/tissue.png" width="150" height="150"></td><td>Perimeter = 569.0437 Area = 7575 Compactness = 42.7472 Orientation = 34.8595 Circularity = 0.29328</td></tr>

  </table>
  </p>

  <h3> Discussion </h3>
  <p>
  For preprocessing, I use dilation and erosion to remove noises and filling
  holes in an object. Our labeling algorithm becomes slow when there are too many
  objects in the image. So, I filter small objects and erosions.
  </p>

  <hr>
  <h2> Segmentation </h2>
  <h3> Problem Definition </h3>
  <p>
  Given frames in a video, I try to detect, segment and track certain object
  in the frames (e.g. Task 1: hand, task 2: bat, or task 3: people) with
  methods we learned from classes.
  </p>

  <h3> Method and Implementation </h3>
  <ol>
    <li>
    <p>
    For the piano dataset,I first found out the mean frame of the image set. Then I subtracted each frame
    from the mean frame . This left me with only the parts with movement in the image. Now to separate the hands
    from the rest of the part I use thresholding and skin color detection.Now I know that the hands are on the left
    part of the image.So I can just focus on the left part and use the cv2 contour function. Now I take the largest
    two contours (which denote the hand) and draw bounding boxes around them.
    </p>
    </li>
    <li>
    <p>
    For the bat dataset,we start with grayscaling the image. After that I apply adaptive thresholding to the image to identify the
    bats present in the image.The I use the opencv contour method and drew bouding boxes around contours with large enough area. To identify the status of the flight/ wing position I use the circularity and compactness.
    </p>
    </li>
    <li>
    <p>
    To detect people in the videos I followed frame subtraction approach. I then use the mean frame
    for background subtraction.After this we used a similar thresholding ,specifically in the 60-255 range. This was followed
    by opening the image. I then used opencv's contour function and cap the area limit on the contours we accept. I found the appropriate through
    experimentation. I then just draw bounding boxes around the remaining contours.
    <table>
    <br>
    <br>
    <tr>
      <td> Examples </td><td> Source </td><td> Segmented </td>
    </tr>
    <tr>
      <td> Piano </td>
      <td> <img src="images/Piano_frame.jpg" width="400" height="280"> </img></td>
      <td> <img src="images/Piano_hands.jpg" width="400" height="280">  </img></td>

    </tr>
    <tr>
      <td> Bat </td>
      <td> <img src="images/bat_frame.jpg" width="400" height="400"> </img></td>
      <td> <img src="images/adaptive_bats.png" width="400" height="400"> </img></td>
    </tr>
    <tr>
      <td> Pedestrian </td>
      <td> <img src="images/Pedestrian_frame.jpg" width="400" height="280"></td>
      <td> <img src="images/Pedestrian_bounding.jpg" width="400" height="280"></td>

    </tr>
  </table>

  <h3> Discussion </h3>
  <p>
  The piano dataset is a bit challenging because of the changes in lighting and shadows within the frames.
  Also the frames are not consecutive so straightforward background subtraction is not very feasible. To get around
  these hurdles, first I calculated the mean frame of the image set. After that I subtract each frame from the mean frame.
  This removes all the static part of the images and what is left are some parts of the piano and pianist. Now to separate out
  the hands , I used skin color detection using the RGB thersholds given in the lab session. After that I could use contouring to find the
  hands in the left part of the image.
  </p>
  <p>
  In the bat dataset, as background changes, it is better to use adpative
  thresholing. If bat is too small or far from a camera, it is hard to tell if
  it is unfolded and there are multiple bats in the region.  I set threshold
  for determining if it is folded emprically with circularity, so that the
  system does not detect the foldness very well for small obejcts
  </p>
  <p>
  In the pedestrian dataset, I tried out multiple models to find the bounding
  boxs for people in all the frames. However the best approach turned out to be straightforward frame subtraction.
  I subtracted each frame from mean frame and found contours on the resulting image. I still could not get around the
  people being occluded by the pole though as it's not present in the mean frame.Drawing bouding boxes on contours that are big enough work very well.
  </p>
  <hr>
  <h2> Conclussion </h2>
  <p>
  I first implemented our own version of getting the first/second moments and
  circularity of objects. I implemented the sequential labeling and boundary
  following algorithm. I applied these algorithms and methods to the given
  four images.
  </p>
  <p>
  For segmenting out hands, our methology works very well at least for the image set we have. Identifying the ROI to be
  on the left side of the image post thersholding and subtraction may not generalize but works well for this.
  </p>
  <p>
  For the bat data set, I found that adaptive thresholding works very well on
  the chages of backgrounds. I detected most of bats except very small bats.
  I also successfully detect whether it is folded with a appropriate threshold
  on circularity, and regions where multiple bats exist.
  </p>
  <p>
  For the person detector task, I used background subtraction and are able to
  identify the majority of people given no occlusion. Occlusion may still
  majorly affect the performance of the model even with my fine tuned
  parameters.
  </p>
  <hr>
  <h2> Credits</h2>
  <ul>
    <li>https://opencv-python-tutroals.readthedocs.io/en/latest/</li>
  </ul>
</div>
</body>
</html>
