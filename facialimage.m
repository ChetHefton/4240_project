sourceimage = imread('lukasource.jpg');
testimage = imread('testluka5.png');

[height, width, ~] = size(sourceimage);
if width > 320
    sourceimage = imresize(sourceimage, [320 NaN]);
end

% Ensure the test image is resized similarly if its width exceeds 320
[widthTest, ~] = size(testimage);

if widthTest > 320
    testimage = imresize(testimage, [320 NaN]);
end

% convert to grayscale
sourceGray = rgb2gray(sourceimage);
testGray   = rgb2gray(testimage);

% face detection
facedetection = vision.CascadeObjectDetector();

location = step(facedetection, sourceimage);
testlocation = step(facedetection, testimage);

deteced_Image = insertShape(sourceimage, 'rectangle', location);
testdeteced_Image = insertShape(testimage, 'rectangle', testlocation);

figure;
imshow(deteced_Image);
title('Detected Faces');

figure;
imshow(testdeteced_Image);
title('Detected Faces in Test Image');

% crop images 
face1 = imcrop(sourceGray, location(1,:));
face2 = imcrop(testGray, testlocation(1,:));

%resize crop
face1 = imresize(face1, [200 200]);
face2 = imresize(face2, [200 200]);

% detect ORB features
points1 = detectORBFeatures(face1);
points2 = detectORBFeatures(face2);

% extract ORB features for the image
[features1, validPoints1] = extractFeatures(face1, points1);
[features2, validPoints2] = extractFeatures(face2, points2);

% Match features between the two images
indexPairs = matchFeatures(features1, features2, ...
    'MatchThreshold', 100,'MaxRatio', 0.8);
% Retrieve matched points
matchedPoints1 = validPoints1(indexPairs(:, 1));
matchedPoints2 = validPoints2(indexPairs(:, 2));

numPoints1 = length(points1);
numPoints2 = length(points2);

numberMatches = size(indexPairs, 1);

% Compute similarity ratio
matchRatio = numberMatches / min(numPoints1, numPoints2);

fprintf('Matched Features: %d\n', numberMatches);
fprintf('Match Ratio: %.2f\n', matchRatio);

% Better threshold
ratioThreshold = 0.11;

if matchRatio >= ratioThreshold
    fprintf("RESULT: Faces MATCH.\n");
else
    fprintf("RESULT: Faces DO NOT MATCH.\n");
end

% visualize matches
figure;
showMatchedFeatures(face1, face2, matchedPoints1, matchedPoints2);
title('Matched ORB Features');

figure;
imshow(face1);
hold on;
plot(points1);
title('ORB Points Face 1');

figure;
imshow(face2);
hold on;
plot(points2);
title('ORB Points Face 2');
