clear; clc; close all;

knownFolder = 'known';
inputFolder = 'input';

%setup detector
faceDetector = vision.CascadeObjectDetector('MergeThreshold', 5);

%define image size for PCA
imgSize = [100 100];
numPixels = prod(imgSize);

knownFiles = dir(fullfile(knownFolder, '*.jpg'));
numImages = length(knownFiles);

%initialize massive matrix
Gamma = zeros(numPixels, numImages);
labels = strings(numImages, 1);

fprintf('Building Matrix from Known Faces...\n');

validCount = 0;
for i = 1:numImages
    fileName = knownFiles(i).name;
    filePath = fullfile(knownFolder, fileName);
    
    img = imread(filePath);
    faceCrop = detectAndCropFace(img, faceDetector);
    
    if isempty(faceCrop)
        fprintf('  No face found in known image: %s\n', fileName);
        continue;
    end
    
    validCount = validCount + 1;
    processedFace = preprocessFace(faceCrop, imgSize);
    
    %flatten 2D image to 1D column vector and store in gamma
    Gamma(:, validCount) = double(processedFace(:));
    labels(validCount) = getLabelFromFilename(fileName, 'Real');
    
    fprintf('  Added %-20s -> label: %s\n', fileName, labels(validCount));
end

%trim unused columns if faces weren't detected
Gamma = Gamma(:, 1:validCount);
labels = labels(1:validCount);
numImages = validCount;

if numImages == 0
    error('No valid known faces were loaded.');
end

fprintf('\nRunning Principal Component Analysis...\n');

%calc mean face
Psi = mean(Gamma, 2);

%sub mean face from all
A = Gamma - Psi;

%conversion matrix trick (L = A' * A)
L = A' * A;

%find eigenvectors V and eigenvalues D of L
[V, D] = eig(L);

%sort eigenvectors by largest eigenvalues
eigenvalues = diag(D);
[~, sortIdx] = sort(eigenvalues, 'descend');
V = V(:, sortIdx);

%map eigenvectors back to original high-dimensional space
U = A * V;

%normalize the eigenfaces
for i = 1:size(U,2)
    U(:,i) = U(:,i) / norm(U(:,i));
end

figure('Name', 'PCA Visualizations');
subplot(1,2,1); imshow(uint8(reshape(Psi, imgSize))); title('Mean Face');
subplot(1,2,2); 
topFace = reshape(U(:,1), imgSize);
imshow(mat2gray(topFace)); title('Top Eigenface');
drawnow;

%project known faces into face space
Omega = U' * A;

fprintf('PCA complete, Face Space generated with %d dimensions.\n\n', numImages);

inputFiles = dir(fullfile(inputFolder, '*.jpg'));

%pre-normalize the known faces for cosine distance
Omega_norm = Omega ./ vecnorm(Omega, 2, 1);

for i = 1:length(inputFiles)
    fileName = inputFiles(i).name;
    filePath = fullfile(inputFolder, fileName);
    
    img = imread(filePath);
    faceCrop = detectAndCropFace(img, faceDetector);
    
    fprintf('\nInput image: %s\n', fileName);
    
    if isempty(faceCrop)
        fprintf('  No face detected.\n');
        continue;
    end
    
    processedFace = preprocessFace(faceCrop, imgSize);
    
    Gamma_test = double(processedFace(:));
    Phi_test = Gamma_test - Psi;
    Omega_test = U' * Phi_test;
    
    %cosine distance calcs
    %normalize the test vector
    Omega_test_norm = Omega_test / norm(Omega_test);
    
    %calculate cosine similarity/dot product
    cosSim = Omega_norm' * Omega_test_norm; 
    
    %convert similarity to distance 
    dists = 1 - cosSim;
    
    [sortedDists, idx] = sort(dists);
    bestDist = sortedDists(1);
    bestLabel = labels(idx(1));
    
    %confidence is natively 0 to 100% now because max distance is roughly 1.0
    confidence = max(0, min(100, 100 * (1 - bestDist)));
    
    if confidence >= 50
        result = "KNOWN";
    else
        result = "UNKNOWN";
    end
    
    fprintf('  Closest match: %s\n', bestLabel);
    fprintf('  Distance: %.4f\n', bestDist);
    fprintf('  Confidence: %.1f%%\n', confidence);
    fprintf('  Result: %s\n', result);
    
    showResult(img, bestLabel, bestDist, confidence, result, i+1);
end

function faceCrop = detectAndCropFace(img, faceDetector)
    bbox = step(faceDetector, img);
    if isempty(bbox)
        faceCrop = []; return;
    end
    
    areas = bbox(:,3) .* bbox(:,4);
    [~, idx] = max(areas);
    bestBox = bbox(idx,:);
    
    shrinkX = bestBox(3) * 0.15; 
    shrinkY = bestBox(4) * 0.15; 
    tightBox = [bestBox(1) + shrinkX, bestBox(2) + shrinkY, ...
                bestBox(3) - (2*shrinkX), bestBox(4) - (2*shrinkY)];
                
    faceCrop = imcrop(img, tightBox);
end

function out = preprocessFace(img, targetSize)
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, targetSize);
    img = adapthisteq(img);
    out = img;
end

function label = getLabelFromFilename(fileName, suffixToRemove)
    [~, name, ~] = fileparts(fileName);
    label = string(name);
    label = erase(label, suffixToRemove);
    label = lower(label);
end

function showResult(img, bestLabel, bestDist, confidence, result, figNum)
    figure(figNum);
    imshow(img);
    title(sprintf('Match: %s | Dist: %.4f | Conf: %.1f%% | %s', ...
        bestLabel, bestDist, confidence, result));
    drawnow;
end