function result = recognizeFace(faceCrop, model)
processedFace = preprocessFace(faceCrop, model.imgSize);

%calculate pca weights
Gamma_test = double(processedFace(:));
Phi_test = Gamma_test - model.Psi;
Omega_test = model.U' * Phi_test;

%extract hog features
hog_test = extractHOGFeatures(processedFace)';

%apply l2 normalization
Omega_test_norm = Omega_test / (norm(Omega_test) + 1e-6);
hog_test_norm = hog_test / (norm(hog_test) + 1e-6);

%fuse and normalize test vector
Fusion_test = [Omega_test_norm; hog_test_norm];

if norm(Fusion_test) == 0
    result.bestLabel = "none";
    result.bestDist = 1;
    result.confidence = 0;
    result.status = "UNKNOWN";
    return;
end

Fusion_test_norm = Fusion_test / norm(Fusion_test);

%calculate cosine similarity against fusion matrix
cosSim = model.FusionMatrix_norm' * Fusion_test_norm;
dists = 1 - cosSim;

[sortedDists, idx] = sort(dists);

bestDist = sortedDists(1);
bestLabel = model.labels(idx(1));

%calculate confidence
confidence = max(0, min(100, 100 * (1 - bestDist)));

if confidence >= 50
    status = "KNOWN";
else
    status = "UNKNOWN";
end

result.bestLabel = bestLabel;
result.bestDist = bestDist;
result.confidence = confidence;
result.status = status;
end