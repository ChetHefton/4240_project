function out = preprocessFace(img, targetSize)
%convert to grayscale if rgb
if size(img,3) == 3
    img = rgb2gray(img);
end

%resize and apply histogram equalization
img = imresize(img, targetSize);
img = adapthisteq(img);

out = img;
end