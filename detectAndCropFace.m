function faceCrop = detectAndCropFace(img, faceDetector)
%run cascade detector
bbox = step(faceDetector, img);

if isempty(bbox)
    faceCrop = [];
    return;
end

%find the largest bounding box if multiple are detected
areas = bbox(:,3) .* bbox(:,4);
[~, idx] = max(areas);
bestBox = bbox(idx,:);

%shrink box slightly for tighter face crop
shrinkX = bestBox(3) * 0.15;
shrinkY = bestBox(4) * 0.15;

tightBox = [bestBox(1) + shrinkX, bestBox(2) + shrinkY, ...
    bestBox(3) - (2 * shrinkX), ...
    bestBox(4) - (2 * shrinkY)];
    
faceCrop = imcrop(img, tightBox);
end