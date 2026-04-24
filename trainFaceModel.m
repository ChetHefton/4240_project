function model = trainFaceModel(knownFolder, faceDetector, imgSize)
people = getKnownPeople(knownFolder);
Gamma = [];
HogMatrix = [];
labels = strings(0);

fprintf('Building Matrix from Known Faces...\n');

%loop through all known people
for i = 1:length(people)
    personName = people(i);
    personFolder = fullfile(knownFolder, personName);
    imgFiles = dir(fullfile(personFolder, '*.jpg'));
    
    %read each image for the person
    for j = 1:length(imgFiles)
        imgPath = fullfile(personFolder, imgFiles(j).name);
        img = imread(imgPath);
        
        faceCrop = detectAndCropFace(img, faceDetector);
        if isempty(faceCrop)
            fprintf('  No face found in known image: %s\n', imgPath);
            continue;
        end
        
        processedFace = preprocessFace(faceCrop, imgSize);
        Gamma = [Gamma, double(processedFace(:))];
        
        %extract hog features as column vector
        hogVector = extractHOGFeatures(processedFace)';
        HogMatrix = [HogMatrix, hogVector];
        
        labels(end+1) = personName;
        fprintf('  Added %s -> label: %s\n', imgPath, personName);
    end
end

numImages = size(Gamma, 2);
if numImages == 0
    model = [];
    fprintf('No valid faces found for training.\n');
    return;
end

fprintf('\nRunning Principal Component Analysis...\n');
Psi = mean(Gamma, 2);
A = Gamma - Psi;
L = A' * A;
[V, D] = eig(L);

eigenvalues = diag(D);
[~, sortIdx] = sort(eigenvalues, 'descend');
V = V(:, sortIdx);

U = A * V;

%normalize eigenvectors
for i = 1:size(U,2)
    if norm(U(:,i)) ~= 0
        U(:,i) = U(:,i) / norm(U(:,i));
    end
end

Omega = U' * A;

%independent l2 normalization
%normalize pca features
Omega_norm = Omega ./ vecnorm(Omega, 2, 1);
%catch zero-vectors
Omega_norm(isnan(Omega_norm)) = 0; 

%normalize hog features
HogMatrix_norm = HogMatrix ./ vecnorm(HogMatrix, 2, 1);
HogMatrix_norm(isnan(HogMatrix_norm)) = 0;

%feature fusion
%stack them vertically. since both have a norm of 1, they carry exactly 50% weight
FusionMatrix = [Omega_norm; HogMatrix_norm];

%normalize the final combined matrix
FusionMatrix_norm = FusionMatrix ./ vecnorm(FusionMatrix, 2, 1);

%save to model struct
model.Psi = Psi;
model.U = U;
model.FusionMatrix_norm = FusionMatrix_norm; 
model.labels = labels;
model.imgSize = imgSize;
model.meanFace = reshape(Psi, imgSize);
model.topEigenface = reshape(U(:,1), imgSize);

fprintf('PCA + HOG Fusion complete. Loaded %d known face images.\n\n', numImages);
end