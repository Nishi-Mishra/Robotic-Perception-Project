import numpy as np
import cv2 as cv
import math
from operator import itemgetter

# Open a video stream.
# Replace 0 with a filename (e.g. 'video.mp4') to read from a video file.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

square_side_length = 35 #in millimeters

#camera intrinsics:
# Optical Sensor: (480, 620)
focal_length = 5.29635 #mm
sensor_width = 7
sensor_height = 7
image_width = 1
image_height = 1

board_4_length = 35*4
board_5_length = 35*5

# Check if the video stream is opened successfully

# Capture a single frame
distance = np.zeros(31)
count = 0
run = True
first = True
while run:
    print('0')
    cap = cv.VideoCapture(0) #switch number to switch camera input
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        run = False
    print('1')
    objp = np.zeros((5*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    ret, frame = cap.read()    
    print('2')
    if ret:
        count += 1
        # Save the frame as an image. You can choose any file type (e.g., '.jpg', '.png')
        #cv.imwrite('saved_frame.jpg', frame)
        print("Frame saved successfully!")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.imshow("preview", gray)
        #cv.imwrite('saved_frame_gray.jpg', gray)
        ret, corners = cv.findChessboardCorners(gray, (5, 4), None)
        if ret == True:
            objpoints.append(objp)
     
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
     
            # Draw and display the corners
            cv.drawChessboardCorners(frame, (5,4), corners2, ret)
            #cv.imshow('img', frame)
            cv.waitKey(500)
            print(len(corners2))
            print(corners[0])
            outer_corners = [corners2[0][0], corners2[4][0], corners2[15][0], corners2[19][0]]
            print(outer_corners)
            bottom_to_top_corners = sorted(outer_corners, key=itemgetter(0))
            left_to_right_corners = sorted(outer_corners, key=itemgetter(1))
            right_edge_pxls = math.dist(left_to_right_corners[2], left_to_right_corners[3])
            left_edge_pxls = math.dist(left_to_right_corners[0], left_to_right_corners[1])
            top_edge_pxls = math.dist(bottom_to_top_corners[2], bottom_to_top_corners[3])
            bottom_edge_pxls = math.dist(bottom_to_top_corners[0], bottom_to_top_corners[1])  
    
            #assuming the vertical edges are parallel to the camera
            image_height_px = frame.shape[0]
            image_width_px = frame.shape[1]
            right_edge_distance = 175 / ((right_edge_pxls/image_height_px*sensor_height)/focal_length) 
            left_edge_distance = 175 / ((left_edge_pxls/image_height_px*sensor_height)/focal_length) 
            top_edge_distance = 140 / ((top_edge_pxls/image_width_px*sensor_width)/focal_length)
            bottom_edge_distance = 140 / ((bottom_edge_pxls/image_width_px*sensor_width)/focal_length)
            right_left_center = (right_edge_distance + left_edge_distance)/2
            top_bottom_center = (top_edge_distance + bottom_edge_distance)/2
            avg_center_distance = (right_left_center + top_bottom_center)/2
            angle = math.pi - math.atan((left_edge_distance - right_edge_distance)/140)
            if first == True:
                start_distance = avg_center_distance
                current_distance = start_distance
                start_y = avg_center_distance * math.cos(( 90 - ((avg_center_distance**2 + right_edge_distance**2 + 70**2) / (2 * avg_center_distance * 70))))
                start_x = avg_center_distance * math.sin(( 90 - ((avg_center_distance**2 + right_edge_distance**2 + 70**2) / (2 * avg_center_distance * 70))))
                print('Origin Pont Set at ' + str(start_distance) + 'mm away')
                print('x: ' + str(start_x))
                print('y: ' + str(start_y))
                start_angle = angle
                print("Angle = " + str(start_angle))
            else:
                current_y = avg_center_distance * math.cos(( 90 - ((avg_center_distance**2 + right_edge_distance**2 + 70**2) / (2 * avg_center_distance * 70))))
                change_y = current_y - start_y
                current_x = avg_center_distance * math.sin(( 90 - ((avg_center_distance**2 + right_edge_distance**2 + 70**2) / (2 * avg_center_distance * 70))))
                change_x = current_x - start_x
                print('Origin Pont Set at ' + str(start_distance) + 'mm away')
                print('x: ' + str(current_x))
                print('y: ' + str(current_y))
                current_angle = angle
                print("Angle = " + str(current_angle - start_angle))
                break
        else:
            print('Could not find checkerboard pattern')
    else:
        print("Error: Could not read a frame from the video stream.")
        run = False

cap.release()
cv.destroyAllWindows()

# Release the capture when done
#cap.release()
#cv.destroyWindow("preview")