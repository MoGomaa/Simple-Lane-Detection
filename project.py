import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def region_of_interest(img, vertices) :
# Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    try :
        channel_count = img.shape[2]
    except IndexError :
        channel_count = 1
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def merge_lines(hough_lines , image_height) :
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            
            if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
                continue
            
            if slope <= 0:              # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

        
    min_y = int(image_height * (3 / 5))
    max_y = image_height

    lines = []
    try :
        poly_left = np.poly1d(np.polyfit(left_line_y,left_line_x,deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        lines.append([left_x_start, max_y, left_x_end, min_y])
    except :
        pass
    try :
        poly_right = np.poly1d(np.polyfit(right_line_y,right_line_x,deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        lines.append([right_x_start, max_y, right_x_end, min_y])
    except :
        pass

    return lines

def draw_lines(image, lines, color=[255, 0, 0], thickness=5, ready=1) :
    line_image = np.zeros_like(image)
    if lines is None:
        return
    if ready == 1 :
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    elif ready == 0 :
        for line in lines :
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image
    
def pipeline(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 100
    high_threshold = 200
    cannyed_image = cv2.Canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    region_of_interest_vertices = np.array([[(0,imshape[0]) , (450,320) , (500, 320) , (imshape[1],imshape[0])]], dtype=np.int32)
    cropped_image = region_of_interest(cannyed_image, region_of_interest_vertices)

    hough_lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    lines = merge_lines(hough_lines , image.shape[0])
    lines_image = draw_lines(image, lines)

    fill_image = np.copy(lines_image)
    try :
        region_of_interest_vertices = np.array([[(lines[0][0], lines[0][1]) , (lines[0][2], lines[0][3]) , (lines[1][2], lines[1][3]) , (lines[1][0], lines[1][1])]], dtype=np.int32)  
        fill_image = region_of_interest(image, region_of_interest_vertices)
        cv2.fillPoly(fill_image, region_of_interest_vertices, 50)
        fill_image = fill_image + lines_image

    except :
        pass

    final_image = np.copy(image)
    final_image = cv2.addWeighted(final_image, 0.8, fill_image, 1.0, 0.0)
    return final_image


cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(video_shape)

output = 'test_videos_output/solidWhiteRight_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output, fourcc, fps, video_shape)
while True :
    ret, frame = cap.read()
    if cv2.waitKey(1)&0xFF == ord('q') or ret == False :
        break
    out_frame = pipeline(frame)
    cv2.imshow("Output", out_frame)
    out.write(out_frame)
    
cap.release()

















    
