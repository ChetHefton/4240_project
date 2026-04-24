function faces = saveDetectedFaces(frame, bbox, inputFolder)
%clear out old input images
delete(fullfile(inputFolder, '*.jpg'));

faces = {};

if isempty(bbox)
    return;
end

%crop and save detected faces
for i = 1:size(bbox, 1)
    faceCrop = imcrop(frame, bbox(i,:));
    
    fileName = sprintf('face_%d.jpg', i);
    filePath = fullfile(inputFolder, fileName);
    
    imwrite(faceCrop, filePath);
    
    faces{end+1} = faceCrop;
end
end