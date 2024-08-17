% Load the intrinsic parameters from the text file
rs_intrinsics = readmatrix('calib_data/realsense_calib/K_f1370224.txt');
hl2_intrinsics = readmatrix('calib_data/hololens_calib/K_hl2.txt');
img_size = [1080, 1920]

fl_rs = [rs_intrinsics(1,1), rs_intrinsics(2,2)];
fl_hl2 = [hl2_intrinsics(1,1), hl2_intrinsics(2,2)];

pp_rs = [rs_intrinsics(1,3), rs_intrinsics(2,3)];
pp_hl2 = [hl2_intrinsics(1,3), hl2_intrinsics(2,3)];

rs = cameraIntrinsics(fl_rs, pp_rs, img_size);
hl2 = cameraIntrinsics(fl_hl2, pp_hl2, img_size);