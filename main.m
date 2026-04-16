% Turns on the camera, and detects faces

clear all

cam = webcam();
cam.Resolution = '640x480';
video_frame = snapshot(cam);

video_Player = vision.VideoPlayer('Position', [100 100 640 480]);

faceDetector = vision.CascadeObjectDetector();
while true
video_frame = snapshot(cam);
bbox = step(faceDetector, video_frame);
video_frame = insertObjectAnnotation(video_frame, 'rectangle', bbox, 'Face');
step(video_Player, video_frame);
end
