% Load the intrinsic parameters from the text file
rs_intrinsics = readmatrix('calib_data/realsense_calib/K_f1370224.txt');
hl2_intrinsics = readmatrix('calib_data/hololens_calib/K_hl2.txt');
img_size = [1080, 1920];

fl_rs = [rs_intrinsics(1,1), rs_intrinsics(2,2)];
fl_hl2 = [hl2_intrinsics(1,1), hl2_intrinsics(2,2)];

pp_rs = [rs_intrinsics(1,3), rs_intrinsics(2,3)];
pp_hl2 = [hl2_intrinsics(1,3), hl2_intrinsics(2,3)];

rs = cameraIntrinsics(fl_rs, pp_rs, img_size);
hl2 = cameraIntrinsics(fl_hl2, pp_hl2, img_size);

% If there are camera parameters, then export the matrix
% Check if 'stereoParams' exists in the workspace
if exist('stereoParams', 'var')
    % If it exists, export the pose_matrix to a txt file
    pose_matrix = stereoParams.PoseCamera2.A;
    pose_matrix(1:3, 4) = pose_matrix(1:3, 4) / 1000;
    disp(pose_matrix)
    writematrix(pose_matrix, 'rs_to_Hl2.txt', 'Delimiter', 'tab');
else
    disp('stereoParams does not exist in the workspace.')
end