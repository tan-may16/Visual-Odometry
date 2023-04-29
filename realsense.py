import pyrealsense2 as rs
import numpy as np
import cv2
    
class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(config)
        

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()
        
        
if __name__ == "__main__":
    
    point = (400, 300)
    def show_distance(event, x, y, args, params):
        global point
        point = (x, y)

    d435i = RealSense()

    # Create mouse event
    cv2.namedWindow("Color frame")
    cv2.setMouseCallback("Color frame", show_distance)
    i=0
    while True:

        ret, depth_frame, color_frame = d435i.get_frame()

        # Show distance for a specific point
        cv2.circle(color_frame, point, 4, (0, 0, 255))
        distance = depth_frame[point[1], point[0]]

        cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("Color frame", color_frame)

        key = cv2.waitKey(100)
        # cv2.imwrite('depth_image'+str(i)+'.png',depth_frame)
        i+=1
        if key == 27:
            break