#!/usr/bin/env python3

import rospy
import os
# watch out on the order for the next two imports lol
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from std_msgs.msg import Float32MultiArray, Float32, String, Int64, Int16
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg  import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math

def load_object_labels(filename):
    """ loads the coco object readable name """

    fo = open(filename,'r')
    lines = fo.readlines()
    fo.close()
    object_labels = {}
    for l in lines:
        object_id = int(l.split(':')[0])
        label = l.split(':')[1][1:].replace('\n','').replace('-','_').replace(' ','_')
        object_labels[object_id] = label

    return object_labels

class DetectorParams:

    def __init__(self, verbose=False):

        # Set to True to use tensorflow and a conv net.
        # False will use a very simple color thresholding to detect stop signs only.
        self.use_tf = rospy.get_param("use_tf")

        # Path to the trained conv net
#        model_path = rospy.get_param("~model_path", "../tfmodels/stop_signs_gazebo.pb")
#        label_path = rospy.get_param("~label_path", "../tfmodels/coco_labels.txt")

#        model_path = rospy.get_param("~model_path", "tfmodels/stop_signs_gazebo.pb")
        model_path = rospy.get_param("~model_path", "tfmodels/ssd_mobilenet_v1_coco.pb")
        label_path = rospy.get_param("~label_path", "tfmodels/coco_labels.txt")
        
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
        self.label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), label_path)

        #Minimum detection distance
        self.detec_min_dist = rospy.get_param("~detec_min_dist", 1.0)

        #Location calculate deviation
        self.detec_loc_dev = rospy.get_param("~detec_loc_dev", 1.0)

        # Minimum score for positive detection
#        self.min_score = rospy.get_param("~min_score", 0.5)
        self.min_score = rospy.get_param("~min_score", 0.8)


        if verbose:
            print("DetectorParams:")
            print("    use_tf = {}".format(self.use_tf))
            print("    model_path = {}".format(model_path))
            print("    label_path = {}".format(label_path))
            print("    min_score = {}".format(self.min_score))

class DetetedObj:

    def __init__(self):
        self.objid = 0
        self.objcategory = 0.
        self.objlocationx = 0.
        self.objlocationy = 0.
        self.robotx = 0.
        self.roboty = 0.
        self.robottheta = 0.
        self.objxlist = []
        self.objylist = []
        self.objinfo = DetectedObject()
        
class Detector:

    def __init__(self):
        rospy.init_node('turtlebot_detector', anonymous=True)
        self.params = DetectorParams()
        self.bridge = CvBridge()

        if self.params.use_tf:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.params.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def,name='')
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)

        # camera and laser parameters that get updated
        self.cx = 0.
        self.cy = 0.
        self.fx = 1.
        self.fy = 1.
        self.laser_ranges = []
        self.laser_angle_increment = 0.01 # this gets updated
        self.x = 0.
        self.y = 0.
#        self.z = 0.
        self.theta = 0.
        self.objx = 0.
        self.objy = 0.
        self.detectedobjinfo = []
        self.detectedobjlist = DetectedObjectList()
        self.robx = 0.
        self.roby = 0.
        self.robtheta = 0.
        self.robxpre = 0.
        self.robypre = 0.
        self.robthetapre = 0.
        self.mode = 0
        self.startcount = 0
        self.robstartx = 0.0
        self.robstarty = 0.0
        self.robstartthe = 0.0

        self.object_publishers = {}
        self.object_labels = load_object_labels(self.params.label_path)

        self.tf_listener = TransformListener()
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback, queue_size=1)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/navigator/robotpos', Pose2D, self.navinfo_callback)
        rospy.Subscriber('/navigator/modeval', Int64, self.modeinfo_callback)

        self.objid_publisher = rospy.Publisher('/obj_rescueid', Float32MultiArray, queue_size=10)
        self.objidlist = []
        self.objid_publisher.publish(data=self.objidlist)

#        self.rescuecommand_publisher = rospy.Publisher('/rescuecommand', Int16, queue_size=10)
#        self.rescuecommand = 0
#        self.rescuecommand_publisher.publish(self.rescuecommand)
        
        self.objlocationpub = rospy.Publisher('/detector/objlocation', Float32MultiArray, queue_size=10)
#        self.debugger2 = rospy.Publisher('/detector/debugger2', String, queue_size=10)
        self.debugger4 = rospy.Publisher('/detector/debugger4', Float32MultiArray, queue_size=10)
        self.pubmarker = rospy.Publisher('detector/marker', Marker, queue_size=10)

    def run_detection(self, img):
        """ runs a detection method in a given image """

        image_np = self.load_image_into_numpy_array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if self.params.use_tf:
            # uses MobileNet to detect objects in images
            # this works well in the real world, but requires
            # good computational resources
#            print("Use tf")
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes,self.d_scores,self.d_classes,self.num_d],
                feed_dict={self.image_tensor: image_np_expanded})

            return self.filter(boxes[0], scores[0], classes[0], num[0])

        else:
            # uses a simple color threshold to detect stop signs
            # this will not work in the real world, but works well in Gazebo
            # with only stop signs in the environment
#            print("Not use tf")
            R = image_np[:,:,0].astype(np.int) > image_np[:,:,1].astype(np.int) + image_np[:,:,2].astype(np.int)
            Ry, Rx, = np.where(R)
            if len(Ry)>0 and len(Rx)>0:
                xmin, xmax = Rx.min(), Rx.max()
                ymin, ymax = Ry.min(), Ry.max()
                boxes = [[float(ymin)/image_np.shape[1], float(xmin)/image_np.shape[0], float(ymax)/image_np.shape[1], float(xmax)/image_np.shape[0]]]
                scores = [.99]
                classes = [13]
                num = 1
            else:
                boxes = []
                scores = 0
                classes = 0
                num = 0

            return boxes, scores, classes, num

    def filter(self, boxes, scores, classes, num):
        """ removes any detected object below MIN_SCORE confidence """

	
        f_scores, f_boxes, f_classes = [], [], []
        f_num = 0

        for i in range(int(num)):
            if scores[i] >= self.params.min_score:
                f_scores.append(scores[i])
                f_boxes.append(boxes[i])
                f_classes.append(int(classes[i]))
                f_num += 1
            else:
                break

        return f_boxes, f_scores, f_classes, f_num

    def load_image_into_numpy_array(self, img):
        """ converts opencv image into a numpy array """

        (im_height, im_width, im_chan) = img.shape

        return np.array(img.data).reshape((im_height, im_width, 3)).astype(np.uint8)

    def drop_marker(self, action, markerid, x, y, name):
        
        if (action==0):
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = rospy.Time.now()
            marker.id = markerid  # enumerate subsequent markers here
            marker.action = Marker.ADD  # can be ADD, REMOVE, or MODIFY
            marker.ns = name
            marker.type = 9   #2

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 5.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.25  # artifact of sketchup export
            marker.scale.y = 0.25  # artifact of sketchup export
            marker.scale.z = 0.25  # artifact of sketchup export

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = name + str(markerid)

            marker.lifetime = rospy.Duration()  # will last forever unless modifie
                        
            self.pubmarker.publish(marker)
        else:
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = rospy.Time.now()
            marker.id = markerid  # enumerate subsequent markers here
            marker.action = Marker.MODIFY  # can be ADD, REMOVE, or MODIFY
            marker.ns = name
            marker.type = 9   #2

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 5.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.25  # artifact of sketchup export
            marker.scale.y = 0.25  # artifact of sketchup export
            marker.scale.z = 0.25  # artifact of sketchup export

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = name + str(markerid)

            marker.lifetime = rospy.Duration()  # will last forever unless modifie
                        
            self.pubmarker.publish(marker)            

    def save_object_list(self,object_msg):
        add_to_list = False
        str = ''
        #when dist < detec_min_dist, consider add it to detectedobjinfo list
#        if (object_msg.distance>0.0) and (object_msg.distance<self.params.detec_min_dist) and (self.mode!=0) and (self.mode!=3):
        if (object_msg.distance>0.0) and (object_msg.distance<self.params.detec_min_dist) and (self.mode!=0) and (self.mode!=1):
            #The detectedobjinfo list is empty
            if len(self.detectedobjinfo)==0:
                newobj = DetetedObj()
                newobj.objid = 1
                newobj.objcategory = object_msg.id
                newobj.objlocationx = self.objx
                newobj.objlocationy = self.objy
                newobj.robotx = self.robx
                newobj.roboty = self.roby
                newobj.robottheta = self.robtheta
                newobj.objinfo = object_msg
                newobj.objxlist.append(self.objx)
                newobj.objylist.append(self.objy)
                #add to list
                self.detectedobjinfo.append(newobj)
                str = str + 'add 1st item'
                
                self.drop_marker(0, newobj.objid, self.objx, self.objy, object_msg.name)
                
            else:
                #The detected obj list is not empty
                str = str + 'found new item'
                for i in range(len(self.detectedobjinfo)):
                    if object_msg.id == self.detectedobjinfo[i].objcategory:
                        #This object category exists in the list:
                        
#                        objxmea = (self.detectedobjinfo[i].objlocationx + self.objx) / 2
#                        objymea = (self.detectedobjinfo[i].objlocationy + self.objy) / 2
                        objxmea = (np.mean(self.detectedobjinfo[i].objxlist) + self.objx) / 2
                        objymea = (np.mean(self.detectedobjinfo[i].objylist) + self.objy) / 2

                        if (self.objx > (objxmea-self.params.detec_loc_dev) and self.objx < (objxmea+self.params.detec_loc_dev)) and (self.objy > (objymea-self.params.detec_loc_dev) and self.objy < (objymea+self.params.detec_loc_dev)):
                            #This item may exists, update this item
                            str = str + 'update exiting item '
                            self.detectedobjinfo[i].objlocationx = self.objx
                            self.detectedobjinfo[i].objlocationy = self.objy
                            self.detectedobjinfo[i].robotx = self.robx
                            self.detectedobjinfo[i].roboty = self.roby
                            self.detectedobjinfo[i].robottheta = self.robtheta
                            self.detectedobjinfo[i].objinfo = object_msg
                            self.drop_marker(1, self.detectedobjinfo[i].objid, self.objx, self.objy, object_msg.name)
                            break
                        else:
                            if i == (len(self.detectedobjinfo)-1):
                                add_to_list = True
                                break                   
                    else:
                        #This object category does not exit in the list, add it to detectedobjinfo list
                        str = str + 'different category '
                        if i == (len(self.detectedobjinfo)-1):                            
                            add_to_list = True
                            break
                    
            if add_to_list:
                newobj = DetetedObj()
                newobj.objid = len(self.detectedobjinfo) + 1
                newobj.objcategory = object_msg.id
                newobj.objlocationx = self.objx
                newobj.objlocationy = self.objy
                newobj.robotx = self.robx
                newobj.roboty = self.roby
                newobj.robottheta = self.robtheta
                newobj.objinfo = object_msg
                newobj.objxlist.append(self.objx)
                newobj.objylist.append(self.objy)
                #add to list
                self.detectedobjinfo.append(newobj)   
                self.drop_marker(0, newobj.objid, self.objx, self.objy, object_msg.name)  

        self.objlocation = []
        self.objlocation.append(len(self.detectedobjinfo)/1.0)
        for i in range(len(self.detectedobjinfo)):
            self.objlocation.append(self.detectedobjinfo[i].objid)
            self.objlocation.append(self.detectedobjinfo[i].objcategory)
            self.objlocation.append(self.detectedobjinfo[i].objlocationx)
            self.objlocation.append(self.detectedobjinfo[i].objlocationy)
            self.objlocation.append(self.detectedobjinfo[i].robotx)
            self.objlocation.append(self.detectedobjinfo[i].roboty)
            self.objlocation.append(self.detectedobjinfo[i].robottheta)
                
        self.objlocationpub.publish(data=self.objlocation)
 #       self.debugger2.publish(str)  

    def project_pixel_to_ray(self, u, v):
        """ takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame """

        ########## Code starts here ##########
        # TODO: Compute x, y, z.

#        x = ((u-self.cx)/self.fx)
#        y = ((v-self.cy)/self.fy)
#        z = 1.

        x = (u - self.cx)/self.fx
        y = (v - self.cy)/self.fy
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm

        ########## Code ends here ##########

        return x, y, z

    def estimate_distance(self, thetaleft, thetaright, ranges):
        """ estimates the distance of an object in between two angles
        using lidar measurements """

        leftray_indx = min(max(0,int(thetaleft/self.laser_angle_increment)),len(ranges))
        rightray_indx = min(max(0,int(thetaright/self.laser_angle_increment)),len(ranges))

        if leftray_indx<rightray_indx:
            meas = ranges[rightray_indx:] + ranges[:leftray_indx]
        else:
            meas = ranges[rightray_indx:leftray_indx]

        num_m, dist = 0, 0
        for m in meas:
            if m>0 and m<float('Inf'):
                dist += m
                num_m += 1
        if num_m>0:
            dist /= num_m

#        self.debugger = rospy.Publisher('/detector/debugger', Float32MultiArray, queue_size=10)
#        self.debugger.publish(data=ranges)

#        self.debugger1 = rospy.Publisher('/detector/debugger1', Int64, queue_size=10)
#        self.debugger1.publish(num_m)
        
#        self.debugger2 = rospy.Publisher('/detector/debugger2', Float32, queue_size=10)
#        self.debugger2.publish(dist)
        
        return dist

    def camera_callback(self, msg):
        """ callback for camera images """

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        (img_h,img_w,img_c) = img.shape

        # runs object detection in the image
        (boxes, scores, classes, num) = self.run_detection(img)

        if num > 0:
            # some objects were detected
            for (box,sc,cl) in zip(boxes, scores, classes):
                ymin = int(box[0]*img_h)
                xmin = int(box[1]*img_w)
                ymax = int(box[2]*img_h)
                xmax = int(box[3]*img_w)
                xcen = int(0.5*(xmax-xmin)+xmin)
                ycen = int(0.5*(ymax-ymin)+ymin)

                cv2.rectangle(img_bgr8, (xmin,ymin), (xmax,ymax), (255,0,0), 2)

                # computes the vectors in camera frame corresponding to each sides of the box
                rayleft = self.project_pixel_to_ray(xmin,ycen)
                rayright = self.project_pixel_to_ray(xmax,ycen)
		
                # convert the rays to angles (with 0 poiting forward for the robot)
                thetaleft = math.atan2(-rayleft[0],rayleft[2])
                thetaright = math.atan2(-rayright[0],rayright[2])
                
                if thetaleft<0:
                    thetaleft += 2.*math.pi
                if thetaright<0:
                    thetaright += 2.*math.pi

                # estimate the corresponding distance using the lidar
                dist = self.estimate_distance(thetaleft,thetaright,img_laser_ranges)

#                if not self.object_publishers.has_key(cl):
                self.object_publishers[cl] = rospy.Publisher('/detector/'+self.object_labels[cl],
                    DetectedObject, queue_size=10)

                self.discover_publisher = rospy.Publisher('/detector/discover',
                    DetectedObject, queue_size=10)
                    
                # publishes the detected object and its location
                object_msg = DetectedObject()
                object_msg.id = cl
                object_msg.name = self.object_labels[cl]
                object_msg.confidence = sc
                object_msg.distance = dist
                object_msg.thetaleft = thetaleft
                object_msg.thetaright = thetaright
                object_msg.corners = [ymin,xmin,ymax,xmax]
                               
                # Get robot position
                rospy.Subscriber('/odom', Odometry, self.cmd_pose_callback)                
                
                self.objlocationdeb = []
                self.objlocationdeb.append(thetaleft)
                self.objlocationdeb.append(thetaright)
                
#                thetamea = (thetaleft + thetaright)/2
#                self.objlocation.append(thetamea)
                
                #calculate object position
                self.objlocationdeb.append(cl/1.0)
#                x_base_cam = dist*np.cos(thetamea)
#                y_base_cam = dist*np.sin(thetamea)
                x_base_cam = (dist*np.cos(thetaleft) + dist*np.cos(thetaright))/2
                y_base_cam = (dist*np.sin(thetaleft) + dist*np.sin(thetaright))/2
                theta = np.arctan2(self.y,self.x)
#                self.objx=self.x + x_base_cam*np.cos(self.theta) - y_base_cam*np.sin(self.theta) 
 #               self.objy=self.y + x_base_cam*np.sin(self.theta) + y_base_cam*np.cos(self.theta)
                objxmea = (self.robx + self.robxpre)/2
                objymea = (self.roby + self.robypre)/2
                objthetamea = (self.robtheta + self.robthetapre)/2
                
                self.objx = objxmea + x_base_cam*np.cos(objthetamea) - y_base_cam*np.sin(objthetamea) 
                self.objy = objymea + x_base_cam*np.sin(objthetamea) + y_base_cam*np.cos(objthetamea)   
                              
#                self.objx=self.robx + x_base_cam*np.cos(self.robtheta) - y_base_cam*np.sin(self.robtheta) 
#                self.objy=self.roby + x_base_cam*np.sin(self.robtheta) + y_base_cam*np.cos(self.robtheta)  
#                self.objlocation.append(dist*np.cos(thetamea))
#                self.objlocation.append(dist*np.sin(thetamea))
                self.objlocationdeb.append(self.objx)
                self.objlocationdeb.append(self.objy)
#                self.objlocationdeb.append(self.x)
#                self.objlocationdeb.append(self.y)
#                self.objlocationdeb.append(self.theta)
#                self.objlocation.append(np.arctan2(self.y,self.x))
                self.objlocationdeb.append(dist)
                self.objlocationdeb.append(self.robx)
                self.objlocationdeb.append(self.roby)
                self.objlocationdeb.append(self.robtheta)
                
                #save it to object_msglist
                self.save_object_list(object_msg)
                
                self.debugger4.publish(data=self.objlocationdeb)                
                                
                #Publish to topic
                
                self.object_publishers[cl].publish(object_msg)
                self.discover_publisher.publish(object_msg)

        # displays the camera image
        cv2.imshow("Camera", img_bgr8)
        cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """ extracts relevant camera intrinsic parameters from the camera_info message.
        cx, cy are the center of the image in pixel (the principal point), fx and fy are
        the focal lengths. """

        ########## Code starts here ##########
        # TODO: Extract camera intrinsic parameters.

        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        
#        debug = []
#        debug.append(self.cx)
#        debug.append(self.cy)
#        debug.append(self.fx)
#        debug.append(self.fy)
        
#        self.debugger3 = rospy.Publisher('/detector/debugger3', Float32MultiArray, queue_size=10)
#        self.debugger3.publish(data=debug)
        ########## Code ends here ##########

    def laser_callback(self, msg):
        """ callback for thr laser rangefinder """

        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment

    def cmd_pose_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
#        self.z = msg.pose.pose.position.z
        orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.theta = yaw

    def navinfo_callback(self, msg):
        self.robxpre = self.robx
        self.robypre = self.roby
        self.robthetapre = self.robtheta
        self.robx = msg.x
        self.roby = msg.y
        self.robtheta = msg.theta
        if (self.startcount==0 and msg.x!=0.0):
            self.robstartx = msg.x
            self.robstarty = msg.y
            self.robstartthe = msg.theta
            self.robxpre = msg.x
            self.robypre = msg.y
            self.robthetapre = msg.theta
            self.startcount = self.startcount +1
            
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = rospy.Time.now()
            marker.id = 111  # enumerate subsequent markers here
            marker.action = Marker.ADD  # can be ADD, REMOVE, or MODIFY
            marker.ns = 'start'
            marker.type = 9   #2

            marker.pose.position.x = self.robstartx
            marker.pose.position.y = self.robstarty
            marker.pose.position.z = 5.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # artifact of sketchup export
            marker.scale.y = 0.1  # artifact of sketchup export
            marker.scale.z = 0.1  # artifact of sketchup export

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = 'start'

            print("**********add a marker on start")
            marker.lifetime = rospy.Duration()  # will last forever unless modifie
                        
            self.pubmarker.publish(marker)
            
            
    def modeinfo_callback(self, msg):
        self.mode = msg.data
#        print("current mode",self.mode)

    def run(self):
        tf.disable_v2_behavior()
        rospy.spin()

if __name__=='__main__':
    d = Detector()
    d.run()
