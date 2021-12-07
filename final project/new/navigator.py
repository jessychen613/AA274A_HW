#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String, Float32MultiArray, Int64, Bool
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj, TspSolver, RRTStar
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from asl_turtlebot.msg import DetectedObject

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    CROSS = 4
    STOP = 5
    RESCUE_PAUSE = 6


obj_id_to_name = {0: 'unlabeled', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush', 92: 'banner', 93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building-other', 97: 'bush', 98: 'cabinet', 99: 'cage', 100: 'cardboard', 101: 'carpet', 102: 'ceiling-other', 103: 'ceiling-tile', 104: 'cloth', 105: 'clothes', 106: 'clouds', 107: 'counter', 108: 'cupboard', 109: 'curtain', 110: 'desk-stuff', 111: 'dirt', 112: 'door-stuff', 113: 'fence', 114: 'floor-marble', 115: 'floor-other', 116: 'floor-stone', 117: 'floor-tile', 118: 'floor-wood', 119: 'flower', 120: 'fog', 121: 'food-other', 122: 'fruit', 123: 'furniture-other', 124: 'grass', 125: 'gravel', 126: 'ground-other', 127: 'hill', 128: 'house', 129: 'leaves', 130: 'light', 131: 'mat', 132: 'metal', 133: 'mirror-stuff', 134: 'moss', 135: 'mountain', 136: 'mud', 137: 'napkin', 138: 'net', 139: 'paper', 140: 'pavement', 141: 'pillow', 142: 'plant-other', 143: 'plastic', 144: 'platform', 145: 'playingfield', 146: 'railing', 147: 'railroad', 148: 'river', 149: 'road', 150: 'rock', 151: 'roof', 152: 'rug', 153: 'salad', 154: 'sand', 155: 'sea', 156: 'shelf', 157: 'sky-other', 158: 'skyscraper', 159: 'snow', 160: 'solid-other', 161: 'stairs', 162: 'stone', 163: 'straw', 164: 'structural-other', 165: 'table', 166: 'tent', 167: 'textile-other', 168: 'towel', 169: 'tree', 170: 'vegetable', 171: 'wall-brick', 172: 'wall-concrete', 173: 'wall-other', 174: 'wall-panel', 175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other', 179: 'waterdrops', 180: 'window-blind', 181: 'window-other', 182: 'wood'}
obj_name_to_id = {name: obj_id for obj_id, name in obj_id_to_name.items()}


class RescueObject:
    def __init__(self, index, obj_id, x, y, tb_x, tb_y, tb_theta):
        self.index = index
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.tb_x = tb_x 
        self.tb_y = tb_y
        self.tb_theta = tb_theta

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE
        self.prev_mode = None  # For printing purposes

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
#        self.v_max = 0.2  # maximum velocity
        self.v_max = 0.08  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)


        # Rescue parameters
        self.rescue = False
        self.rescue_name_list = []
        self.rescue_location_and_name_queue = []  # ((x,y), name) tuple
        rospy.Subscriber("/begin_rescue", Bool, self.rescue_signal_callback)

        # TODO: REMOVE (ONLY FOR DEBUGGING)
        self.replan_publisher = rospy.Publisher("/replan", Bool, queue_size = 10)

        # Object detector: if, in real time, we see the object
        rospy.Subscriber("/detector/discover", DetectedObject, self.object_detected_callback)

        # Object id and location list
        self.object_list = []  # List of RescueObject's
        rospy.Subscriber("/detector/objlocation", Float32MultiArray, self.object_list_callback)

        # Time to stop at a stop sign (or detected objects during rescue mission)
        self.stop_time = rospy.get_param("~stop_time", 5.)

        # Minimum distance from a stop sign to obey it
        # self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.9)        

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        self.robotpospub = rospy.Publisher('/navigator/robotpos', Pose2D, queue_size=10)
        self.currentmode = rospy.Publisher('/navigator/mode', String, queue_size=10)
        self.currentmodeval = rospy.Publisher('/navigator/modeval', Int64, queue_size=10)
        print("finished init")

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}, rescue_objects:{rescue_object_list}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        self.rescue_name_list = config["rescue_object_list"].split(",")
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def rescue_signal_callback(self, msg):
        self.rescue = True
        rospy.loginfo("Beginning rescue mission.")

        # Call TSP planner to get queue of positions to navigate through the different locations

        rescue_locations_list = []

        chosen_object_indices = set()  # set of the indices within the world

        # greedily use the first object that matches the object description

        rescue_loc_idx_to_obj_list_idx = {}

        # find the associated locations for each object in the 
        for obj_name in self.rescue_name_list:
            for obj_list_idx, obj in enumerate(self.object_list):
                if obj.index not in chosen_object_indices and obj.obj_id == obj_name_to_id[obj_name]:
                    rescue_locations_list.append((obj.x, obj.y))

                    rescue_loc_idx_to_obj_list_idx[len(rescue_locations_list) - 1] = obj_list_idx

                    chosen_object_indices.add(obj.index)

                    continue

        # TODO: include theta in positions
        # either have the goal be to reach the location (then once reached, replan, or to detect the object from close then replan)
        tsp_planner = TspSolver(start_location = (self.x, self.y), object_locations = rescue_locations_list)

        location_queue, obj_id_queue = tsp_planner.plan()

        location_and_name_queue = []

        # for idx in range(len(obj_id_queue)):
        #     location_and_name_queue.append((location_queue[idx], obj_id_to_name[obj_id_queue[idx]]))

        # location_and_name_queue.append((location_queue[-1], "STARTING_LOCATION"))

        for idx in range(len(obj_id_queue)):
            rescue_obj = self.object_list[rescue_loc_idx_to_obj_list_idx[obj_id_queue[idx]]]
            location_and_name_queue.append(((rescue_obj.tb_x, rescue_obj.tb_y, rescue_obj.tb_theta), obj_id_to_name[rescue_obj.obj_id]))

        location_and_name_queue.append(((self.x, self.y, self.theta), "STARTING_LOCATION"))

        self.rescue_location_and_name_queue = location_and_name_queue

        self.rescue_next_object()
        # self.x_g = self.rescue_location_and_name_queue[0][0][0]
        # self.y_g = self.rescue_location_and_name_queue[0][0][1]

        # self.replan()


    def object_list_callback(self, msg):
        arr = msg.data
        n_objects = int(arr[0])

        object_list = []

        entires_per_object = 7

        for obj_idx in range(n_objects):
            index = int(arr[entires_per_object*obj_idx + 1])
            obj_id = int(arr[entires_per_object*obj_idx + 2])
            x = arr[entires_per_object*obj_idx + 3]
            y = arr[entires_per_object*obj_idx + 4]
            tb_x = arr[entires_per_object*obj_idx + 5]
            tb_y = arr[entires_per_object*obj_idx + 6]
            tb_theta = arr[entires_per_object*obj_idx + 7]

            object_list.append(RescueObject(index, obj_id, x, y, tb_x, tb_y, tb_theta))

        self.object_list = object_list

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                8,
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def object_detected_callback(self, msg):
        """ callback for when the detector has found an object. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance
        name = msg.name
        confidence = msg.confidence
        thetaleft = msg.thetaleft
        thetaright = msg.thetaright
        corners = msg.corners
        
#        print("detection",name,confidence,dist,thetaleft,thetaright)

        # check to see if the object is the most recent object that we want ot detect

        # TODO: maintain the separate logic for stop signs

        # if close enough and in nav mode, stop
        #if (not self.rescue or msg.name == self.rescue_location_and_name_queue[0][1]) and dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK:
        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK:
            print(f"Object {name} detected, stop.")

            # remove the object from the queue
            #self.rescue_location_and_name_queue.pop(0)
            #self.init_rescue_stop()
            self.init_stop_sign()


    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def init_rescue_pause(self):
        self.rescue_pause_start = rospy.get_rostime()
        self.mode = Mode.RESCUE_PAUSE

    def has_rescue_paused(self):
        return self.mode == Mode.RESCUE_PAUSE and \
            rospy.get_rostime() - self.rescue_pause_start > rospy.Duration.from_sec(self.stop_time)


    def rescue_next_object(self):
        # Load next location in queue if possible

        if len(self.rescue_location_and_name_queue):
            print(f"The next object we're looking to navigate to is {self.rescue_location_and_name_queue[0][1]}")
            self.x_g = self.rescue_location_and_name_queue[0][0][0]
            self.y_g = self.rescue_location_and_name_queue[0][0][1]
            self.theta_g = self.rescue_location_and_name_queue[0][0][2]
            self.replan()
        else:
            # forget about goal:
            self.x_g = None
            self.y_g = None
            self.theta_g = None
            self.switch_mode(Mode.IDLE)


    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        # NOTE: this has been modified to represent stopping at a stop sign and goal rescue objects

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.crossing_time)

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """



        self.replan_publisher.publish(Bool(True))

        if self.mode == Mode.RESCUE_PAUSE and not self.has_rescue_paused():
            return

        print(f"!!!!!!!!!!!Replanning while in mode {self.mode}!!!!!!!!!!!")

        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))

        print(f"!!!!!!!!!!!!!!!!!Goal is {x_goal}!!!!!!!!!!!!!!!!!!!!!")

        #problem = AStar(
        problem = RRTStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Navigator: computing navigation plan!!!!!!!!!!!!!!!!!!!!!!!!")
        success = problem.solve()
        if not success:
            rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!Planning failed!!!!!!!!!!!!!!!!!")
            return
        rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!!!!!!Planning Succeeded!!!!!!!!!!!!!!!!!!!!!!")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("!!!!!!!!!!!!!!!!!!Path too short to track!!!!!!!!!!!!!!!!!!!!!!")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def publish_robot_pos(self):
        pose = Pose2D()
        pose.x = self.x
        pose.y = self.y
        pose.theta = self.theta

        self.robotpospub.publish(pose)  


    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # print("robot x,y,theta",self.x,self.y,self.theta)
            self.publish_robot_pos()
            
            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            
            # logs the current mode
            if self.prev_mode != self.mode:
                rospy.loginfo("Current mode: %s", self.mode)
                self.prev_mode = self.mode
            
            strmode = "Current mode: " + str(self.mode)
            self.currentmode.publish(strmode)  
            self.currentmodeval.publish(int(self.mode.value))
            
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.STOP:
                 # At a stop sign
                if self.has_stopped():
                    self.init_crossing()
            elif self.mode == Mode.RESCUE_PAUSE:
                if self.has_rescue_paused():
                    print("Ended the rescue pause! Now looking to navigate to the next object!")
                    self.rescue_next_object()
            elif self.mode == Mode.CROSS:
                # Crossing an intersection
                if self.has_crossed():
                    self.mode = Mode.TRACK         
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    print(f"Made it to goal! rescue is {self.rescue}")
                    if self.rescue and len(self.rescue_location_and_name_queue):
                        # move to the next location in the rescue location queue
                        rescue_location = self.rescue_location_and_name_queue[0][1]
                        rospy.loginfo(f"Made it to rescue location {rescue_location}.")

                        # pop the most recently found object off of the rescue location queue
                        self.rescue_location_and_name_queue.pop(0)

                        # if there are no objects left in the rescue queue, then idle
                        if not len(self.rescue_location_and_name_queue):
                            print("We've completed the rescue mission!")

                            # forget about goal:
                            self.x_g = None
                            self.y_g = None
                            self.theta_g = None
                            self.switch_mode(Mode.IDLE)

                            self.rescue = False

                        self.init_rescue_pause()
                    else:
                        # forget about goal:
                        self.x_g = None
                        self.y_g = None
                        self.theta_g = None
                        self.switch_mode(Mode.IDLE)

            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
