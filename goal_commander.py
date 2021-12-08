#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import Int16, String, Float32MultiArray,Float32
import tf
import numpy as np
from numpy import linalg

# if using gmapping, you will have a map frame. otherwise it will be odom frame
mapping = True

class GoalPoseCommander:

    def __init__(self):
        rospy.init_node('goal_pose_commander', anonymous=True)
        # initialize variables
        self.x_g = None
        self.y_g = None
        self.theta_g = None
        self.goal_pose_received = False
        self.trans_listener = tf.TransformListener()
        self.start_time = rospy.get_rostime()
        self.command = 0.0
        self.startcount = 0
        self.count = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.x_goal = 0.0
        self.y_goal = 0.0
        self.robstartx = 0.0
        self.robstarty = 0.0
        self.robstartthe = 0.0
        self.objlocationlist = []
        self.objreslocation = []
        self.obj_rescueid = 0.0
        self.debuginfo = ''
        self.reachedgoal = 0.0
        self.index = 0
        self.near_thresh = 0.2
                
        # command pose for controller
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber('/navigator/robotpos', Pose2D, self.navinfo_callback)
        rospy.Subscriber('/rescuecommand', Float32, self.command_callback)

        # Object location
        rospy.Subscriber('/detector/objlocation', Float32MultiArray, self.object_location_callback)
        rospy.Subscriber('/obj_rescueid', Float32, self.object_id_callback)
        rospy.Subscriber('/reachgoal', Float32, self.reach_goal_callback)
       
        self.debugger2 = rospy.Publisher('/goal/debugger2', String, queue_size=10)  
        
    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        rospy.loginfo("rviz command received!")
        try:
            origin_frame = "/map" if mapping else "/odom"
            rospy.loginfo("getting frame")
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (
                    nav_pose_origin.pose.orientation.x,
                    nav_pose_origin.pose.orientation.y,
                    nav_pose_origin.pose.orientation.z,
                    nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
            self.start_time = rospy.get_rostime()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def near_goal(self):

        return (
            linalg.norm(np.array([self.x - self.x_goal, self.y - self.y_goal]))
            < self.near_thresh
        )


    def command_callback(self, msg):
        self.command = msg.data
        strdebug = "command received:" + str(self.command)
        
        self.debugger2.publish(strdebug)  

    def reach_goal_callback(self, msg):
        self.reachedgoal = msg.data
        if (self.command == 2.0):
            self.command = 2.0
        strdebug1 = "reached goal:" + str(self.reachedgoal) + "self.command: " + str(self.command)
        
        self.debugger2.publish(strdebug1)  


    def object_id_callback(self, msg):
        self.obj_rescueid = msg.data
        strdebug2 = "object id received:" + str(self.obj_rescueid)
        
        if (self.obj_rescueid == 1.0):
            strdebug2 = strdebug2 + "****received id 1.0****"
            
#        if (int(self.obj_rescueid) == 1):
#            strdebug2 = strdebug2 + "###transfer int is OK###"
        
        objlen = len(self.objlocationlist)
        objind = (objlen - 1) // 7
        strdebug2 = strdebug2 + "###total records: " + str(objlen) + " " + str(objind)
        if objlen>0:
            for i in range(objind):
                if (self.obj_rescueid == self.objlocationlist[i*7+1]):
                    
                    objreslen = len(self.objreslocation)
                    strdebug2 = strdebug2 + "find a record" + str(self.objlocationlist[7*i+5]) + str(self.objlocationlist[7*i+6]) + str(self.objlocationlist[7*i+7])
                    self.objreslocation.append(self.objlocationlist[7*i+5])
                    self.objreslocation.append(self.objlocationlist[7*i+6])
                    self.objreslocation.append(self.objlocationlist[7*i+7])
                    strdebug2 = strdebug2 + "objreslocation"
                    for i in range(len(self.objreslocation)):                        
                        strdebug2 = strdebug2 + " " + str(self.objreslocation[i]) + " "
#                    break
                
        self.debugger2.publish(strdebug2)  
        print("#####self.objreslocation:", self.objreslocation)

    def object_location_callback(self, msg):
        self.objlocationlist = msg.data
#        strdebug = "object location received:" + str(self.objlocationlist)

#        strdebug = "object location received:" + str(len(self.objlocationlist))
        
#        self.debugger2.publish(strdebug)  

    def navinfo_callback(self, msg):

        if (self.startcount==0 and msg.x!=0.0):
            self.robstartx = msg.x
            self.robstarty = msg.y
            self.robstartthe = msg.theta
            self.startcount = self.startcount +1
        else:
            self.x = msg.x
            self.y = msg.y
            self.theta = msg.theta
        strdebug2 = "Starting point: " + str(self.x) + str(self.y) + str(self.theta)
        self.debugger2.publish(strdebug2)  

#            print("Starting point: ", robstartx, robstarty, robstartthe)

    def publish_goal_pose(self):
        """ sends the current desired pose to the navigator """
        strdebug3 = ""
        
        if (self.command!=0.0):
            if (self.command==1.0):
                strdebug3 = strdebug3 + "back to start "
                pose_g_msg = Pose2D()
                pose_g_msg.x = self.robstartx
                pose_g_msg.y = self.robstarty
                pose_g_msg.theta = self.robstartthe
                strdebug3 = strdebug3 + "Set to Starting point: " + str(self.robstartx) + str(self.robstarty) + str(self.robstartthe)
                self.nav_goal_publisher.publish(pose_g_msg)
                self.command = 0.0
            else:
#                strdebug3 = strdebug3 + "navigate to the goal "
                
                objrelen = len(self.objreslocation)
                strdebug3 = strdebug3 + "item number " + str(len(self.objreslocation)) + "objrelen: " + str(objrelen)
#                self.debugger2.publish(strdebug3) 
                if (self.index<objrelen) and (objrelen>0):
                    strdebug3 = strdebug3 + "objreslocation is not empty , index=" + str(self.index) + " " + str(objrelen)
                    if (self.count == 0):
                        pose_g_msg = Pose2D()
                        pose_g_msg.x = self.objreslocation[self.index]
                        pose_g_msg.y = self.objreslocation[self.index+1]
                        pose_g_msg.theta = self.objreslocation[self.index+2]
                        self.x_goal = self.objreslocation[self.index]
                        self.y_goal = self.objreslocation[self.index+1]
                        strdebug3 = strdebug3 + "Set to goal point: " + str(pose_g_msg.x) + str(pose_g_msg.y) + str(pose_g_msg.theta)
                        self.nav_goal_publisher.publish(pose_g_msg)
                        self.count = self.count +1
#                    self.debugger2.publish(strdebug3) 
                    if self.near_goal():
                        strdebug3 = strdebug3 + "near the to goal, set back" 
                        self.index = self.index +3
                        self.count = 0
                    self.debugger2.publish(strdebug3) 
                else:
                    self.command = 1.0
                    self.index = 0                    
        else:
#            strdebug3 = "rviz click "
            if self.x_g is not None:
                pose_g_msg = Pose2D()
                pose_g_msg.x = self.x_g
                pose_g_msg.y = self.y_g
                pose_g_msg.theta = self.theta_g
                self.nav_goal_publisher.publish(pose_g_msg)
#                self.debugger2.publish(strdebug3) 

        self.debugger2.publish(strdebug3) 
        
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (self.command!=0):
                self.publish_goal_pose()
            else:
                t = rospy.get_rostime()
                if (t - self.start_time).to_sec() < 2.0:
                    self.publish_goal_pose()
            rate.sleep()
        
if __name__ == '__main__':
    sup = GoalPoseCommander()
    try:
        sup.loop()
    except rospy.ROSInterruptException:
        pass        
    
