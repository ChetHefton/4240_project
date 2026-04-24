function main
clc; close all;

knownFolder = 'known';
inputFolder = 'input';
imgSize = [100 100];

%set delay between facial recognition scans in seconds
captureDelay = 1.5;

if ~exist(inputFolder, 'dir')
    mkdir(inputFolder);
end
if ~exist(knownFolder, 'dir')
    mkdir(knownFolder);
end


faceDetector = vision.CascadeObjectDetector('MergeThreshold', 5);

%build the initial pca and hog fusion model on application startup
fprintf('Training face model...\n');
model = trainFaceModel(knownFolder, faceDetector, imgSize);

%setup main figure window
fig = figure('Name', 'Face Security Cam', ...
    'NumberTitle', 'off', ...
    'Position', [100 100 950 620], ...
    'CloseRequestFcn', @closeApp);

%create main axis for live webcam feed
ax = axes('Parent', fig, ...
    'Units', 'pixels', ...
    'Position', [40 120 640 480]);

%create axis for displaying the calculated mean face
meanAx = axes('Parent', fig, ...
    'Units', 'pixels', ...
    'Position', [40 20 150 80]);
title(meanAx, 'Mean Face');

%create axis for displaying the top eigenvector face
eigenAx = axes('Parent', fig, ...
    'Units', 'pixels', ...
    'Position', [200 20 150 80]);
title(eigenAx, 'Top Eigenface');

%setup text label for recognition results
resultText = uicontrol(fig, 'Style', 'text', ...
    'String', 'Result: Waiting...', ...
    'FontSize', 14, ...
    'HorizontalAlignment', 'left', ...
    'Position', [710 500 220 40]);

%setup text label for similarity confidence percentage
confidenceText = uicontrol(fig, 'Style', 'text', ...
    'String', 'Confidence: --', ...
    'FontSize', 12, ...
    'HorizontalAlignment', 'left', ...
    'Position', [710 460 220 35]);

%create ui buttons and link them to their respective callback functions
uicontrol(fig, 'Style', 'pushbutton', ...
    'String', 'Start Camera', ...
    'FontSize', 12, ...
    'Position', [710 380 170 40], ...
    'Callback', @startCamera);

uicontrol(fig, 'Style', 'pushbutton', ...
    'String', 'Stop Camera', ...
    'FontSize', 12, ...
    'Position', [710 330 170 40], ...
    'Callback', @stopCamera);

uicontrol(fig, 'Style', 'pushbutton', ...
    'String', 'Register New Person', ...
    'FontSize', 12, ...
    'Position', [710 250 170 40], ...
    'Callback', @registerPerson);

uicontrol(fig, 'Style', 'pushbutton', ...
    'String', 'View / Delete Person', ...
    'FontSize', 12, ...
    'Position', [710 200 170 40], ...
    'Callback', @deletePerson);

%draw the initial mean and eigenface visuals on startup
updatePCAVisuals();

%initialize global state variables for the camera
cam = [];
running = false;
lastCaptureTime = tic;

    function startCamera(~, ~)
        %prevent starting multiple camera instances
        if running
            return;
        end
        
        %initialize hardware and set resolution
        cam = webcam();
        cam.Resolution = '640x480';
        running = true;
        lastCaptureTime = tic;
        
        set(resultText, 'String', 'Result: Camera running...');
        
        %main processing loop while figure is open
        while running && ishandle(fig)
            %grab current frame and detect faces
            frame = snapshot(cam);
            bbox = step(faceDetector, frame);
            
            %draw bounding box on live feed
            annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bbox, 'Face');
            imshow(annotatedFrame, 'Parent', ax);
            drawnow;
            
            %execute recognition logic based on capture delay
            if toc(lastCaptureTime) >= captureDelay
                faces = saveDetectedFaces(frame, bbox, inputFolder);
                
                %update ui based on detection results
                if isempty(faces)
                    set(resultText, 'String', 'Result: No face');
                    set(confidenceText, 'String', 'Confidence: --');
                else
                    %run the feature fusion classification
                    result = recognizeFace(faces{1}, model);
                    set(resultText, 'String', sprintf('Result: %s - %s', ...
                        result.status, result.bestLabel));
                    set(confidenceText, 'String', sprintf('Confidence: %.1f%%', ...
                        result.confidence));
                end
                
                %reset timer
                lastCaptureTime = tic;
            end
        end
    end

    function stopCamera(~, ~)
        %halt processing loop and update ui
        running = false;
        set(resultText, 'String', 'Result: Camera stopped');
        set(confidenceText, 'String', 'Confidence: --');
        
        %release camera hardware
        if ~isempty(cam)
            clear cam;
            cam = [];
        end
    end

    function registerPerson(~, ~)
        %pause live feed during registration
        wasRunning = running;
        stopCamera();
        
        %prompt user for identification
        nameBox = inputdlg('Enter the person name:', 'Register Person');
        if isempty(nameBox)
            return;
        end
        
        %sanitize input string for folder creation
        personName = lower(strtrim(nameBox{1}));
        if personName == ""
            return;
        end
        personName = regexprep(personName, '[^\w\s-]', '');
        personName = strrep(personName, ' ', '_');
        
        %create storage directory for new subject
        personFolder = fullfile(knownFolder, personName);
        if ~exist(personFolder, 'dir')
            mkdir(personFolder);
        end
        
        %initialize temporary camera for capture process
        camReg = webcam();
        camReg.Resolution = '640x480';
        poses = ["straight", "left", "right", "up", "down"];
        photoNum = 1;
        
        %capture required profile poses
        while photoNum <= 5
            instruction = sprintf( ...
                'Photo %d of 5\nLook slightly %s.\n\nPress OK when ready.', ...
                photoNum, poses(photoNum));
            uiwait(msgbox(instruction, 'Registration'));
            
            goodPhoto = false;
            while ~goodPhoto
                frame = snapshot(camReg);
                bbox = step(faceDetector, frame);
                annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bbox, 'Face');
                
                imshow(annotatedFrame, 'Parent', ax);
                drawnow;
                
                %verify exactly one subject is in frame before saving
                if size(bbox, 1) == 1
                    fileName = sprintf('%s_%d.jpg', personName, photoNum);
                    
                    %save full frame to retain context for training detector
                    imwrite(frame, fullfile(personFolder, fileName));
                    
                    set(resultText, 'String', sprintf('Saved photo %d of 5', photoNum));
                    pause(1);
                    
                    goodPhoto = true;
                    photoNum = photoNum + 1;
                else
                    uiwait(warndlg( ...
                        'Need exactly ONE face detected. Adjust yourself and press OK to retry.', ...
                        'Face Detection'));
                end
            end
        end
        
        %cleanup and trigger model retraining
        clear camReg;
        
        set(resultText, 'String', sprintf('Registered: %s', personName));
        set(confidenceText, 'String', 'Retraining model...');
        
        model = trainFaceModel(knownFolder, faceDetector, imgSize);
        updatePCAVisuals();
        
        set(confidenceText, 'String', 'Model updated');
        
        %resume monitoring if it was previously active
        if wasRunning
            startCamera();
        end
    end

    function deletePerson(~, ~)
        %pause live feed
        wasRunning = running;
        stopCamera();
        
        %retrieve current database subjects
        people = getKnownPeople(knownFolder);
        if isempty(people)
            msgbox('No people saved yet.', 'Known People');
            return;
        end
        
        %display selection menu
        [idx, ok] = listdlg('PromptString', 'Saved people. Select one to delete:', ...
            'SelectionMode', 'single', ...
            'ListString', cellstr(people));
        if ~ok
            return;
        end
        
        %confirm and execute deletion
        personName = people(idx);
        personFolder = fullfile(knownFolder, personName);
        answer = questdlg(sprintf('Delete %s from known database?', personName), ...
            'Confirm Delete', 'Yes', 'No', 'No');
            
        if answer == "Yes"
            rmdir(personFolder, 's');
            set(resultText, 'String', sprintf('Deleted: %s', personName));
            set(confidenceText, 'String', 'Retraining model...');
            
            %check if database is empty before retraining
            peopleLeft = getKnownPeople(knownFolder);
            if isempty(peopleLeft)
                set(confidenceText, 'String', 'No known people left');
            else
                model = trainFaceModel(knownFolder, faceDetector, imgSize);
                updatePCAVisuals();
                set(confidenceText, 'String', 'Model updated');
            end
        end
        
        %resume monitoring
        if wasRunning
            startCamera();
        end
    end

    function closeApp(~, ~)
        %handle safe shutdown of hardware and figure
        running = false;
        if ~isempty(cam)
            clear cam;
        end
        delete(fig);
    end

    function updatePCAVisuals()
        %clear existing plots
        cla(meanAx);
        cla(eigenAx);
        
        %check if model is empty before trying to draw
        if isempty(model)
            title(meanAx, 'Mean Face (Empty)');
            title(eigenAx, 'Top Eigenface (Empty)');
            return;
        end
        
        %extract visualization data from struct
        meanImg = model.meanFace;
        eigenImg = model.topEigenface;
        
        %render mean face
        imshow(mat2gray(meanImg), 'Parent', meanAx);
        title(meanAx, 'Mean Face');
        
        %render top principal component
        imshow(mat2gray(eigenImg), 'Parent', eigenAx);
        title(eigenAx, 'Top Eigenface');
        drawnow;
    end
end