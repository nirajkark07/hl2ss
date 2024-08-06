import numpy as np
import os
import cv2
import glob
from enum import Enum
import json

class DrawOption(Enum):
    AXES = 1
    CUBE = 2

def drawAxes(img, corners, imgpts):
    def tupleofInts(arr):
        return tuple(int(x) for x in arr)
    
    corner = tupleofInts(corners[0].ravel())
    img = cv2.line(img, corner, tupleofInts(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tupleofInts(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tupleofInts(imgpts[2].ravel()), (0,0,255), 5)
    
    return img

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    # Add green plane
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # Add box borders
    for i in range(4):
        j = i + 4
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    
    return img

# Create homogeneous transformation matrix form rvec to tvec
def create_homogeneous_matrix(rvec, tvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Create homogeneous transformation matrix
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = tvec.ravel()
    return H

def poseEstimation(option: DrawOption, directory_path, output_file):
    # Load intrinsic parameters.
    k_path = glob.glob(os.path.join(directory_path, '*.txt'))
    k = np.loadtxt(k_path[0])

    # Passing zero distortion coefficients
    d = np.zeros((5, 1))

    # Read image
    imgPathlist = glob.glob(os.path.join(directory_path, '*.png'))

    # Initialize
    square_size = 0.021 # size of each square in meters
    nRows = 7
    nCols = 10
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)*square_size

    # World points of object to be drawn
    axis = np.float32([[3*square_size,0,0], [0,3*square_size,0],[0,0,-3*square_size]])
    cubeCorners = np.float32([[0,0,0], [0,3*square_size,0], [3*square_size,3*square_size,0],[3*square_size,0,0],[0,0,-3*square_size],[0,3*square_size,-3*square_size],[3*square_size,3*square_size,-3*square_size],[3*square_size,0,-3*square_size]])

    # Blank array to store transformation matricies
    transformation_matricies = []

    # Find corners
    for curImgPath in imgPathlist:
        imgBGR = cv2.imread(curImgPath)
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:
            cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11,11), (-1, -1), termCriteria)
            _, rvecs, tvecs = cv2.solvePnP(worldPtsCur, cornersRefined, k, d)

            # Create homogeneous transformation matrix and add to list
            H = create_homogeneous_matrix(rvecs, tvecs)
            print(H)
            transformation_matricies.append(H.tolist())

            if option == DrawOption.AXES:
                imgpts, _ =  cv2.projectPoints(axis,rvecs,tvecs,k,d)
                imgBGR = drawAxes(imgBGR, cornersRefined,imgpts)

            if option == DrawOption.CUBE:
                imgpts, _ = cv2.projectPoints(cubeCorners, rvecs,tvecs,k,d)
                imgBGR = drawCube(imgBGR, imgpts)
            
            cv2.imshow('Chessboard', imgBGR)
            cv2.waitKey(10000)
        
        with open(os.path.join(directory_path, output_file), "w") as f:
            f.write(f"{H[0][0]}" f"{H[0][1]}" f"{H[0][2]}" f"{H[0][3]}\n")
            f.write(f"{H[1][0]}" f"{H[1][1]}" f"{H[1][2]}" f"{H[1][3]}\n")
            f.write(f"{H[2][0]}" f"{H[2][1]}" f"{H[2][2]}" f"{H[2][3]}\n")
            f.write(f"{H[3][0]}" f"{H[3][1]}" f"{H[3][2]}" f"{H[3][3]}\n")

            return imgBGR

if __name__ == '__main__':
    realsense_dir = r"calib_data/realsense_calib"
    hololens_dir = r"calib_data/hololens_calib"
    rs_output = r"rs_trans.txt"
    hl2_output = r"hl2_trans.txt"

    rs_result = poseEstimation(DrawOption.CUBE, realsense_dir, rs_output)
    hl2_result = poseEstimation(DrawOption.CUBE, hololens_dir, hl2_output)

 
