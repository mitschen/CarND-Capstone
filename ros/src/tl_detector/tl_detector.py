#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import numpy as np
import cv2
import yaml
import math
import sys


STATE_COUNT_THRESHOLD = 3
class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.has_image = False
        self.lights = []
        
        self.stopLineIndex = []
        
        #Temporary variable
        ###########################
        self.globale_counter = 0
        ###########################

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        rospy.logwarn("StopLines {0}".format(len(self.config)) + str(self.config))

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        
        self.counter = 0
        
        

        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg
        return
        wp, self.state = self.process_traffic_lights()
        
        if(self.last_state != self.state):
          self.last_state = self.state
          if(self.state == TrafficLight.UNKNOWN):
            self.upcoming_red_light_pub.publish(-1)
            rospy.loginfo("Publish information {0}".format(-1))
            return
          
        if(self.last_wp != wp):
          self.last_wp = wp
          self.upcoming_red_light_pub.publish(self.last_wp)
          rospy.loginfo("Publish information {0}".format(self.last_wp))
            
            
        
        
    def calcDistance_PoseStamped(self, pose1, pose2):
      return math.sqrt((pose1.pose.position.x-pose2.pose.position.x)**2 
                       + (pose1.pose.position.y-pose2.pose.position.y)**2)
      
    #calculates the waypoint of each stop line
    #and initializes a list of self.stopLineIndex.
    #The stopLineIndex list is sorted according to the waypoints.
    #Each element contains the waypoint-index and the trafficlight index
    def calculate_stopline_index(self):
      rospy.loginfo("calculate_stop_line_index called")
      stopLineArray = self.config['stop_line_positions']
      for i in range(len(stopLineArray)):
        stopLinePose = PoseStamped()
        stopLinePose.pose.position.x = stopLineArray[i][0]
        stopLinePose.pose.position.y = stopLineArray[i][1]
        wpindex = self.get_closest_waypoint(stopLinePose)
        self.stopLineIndex.append( [wpindex, i])
      self.stopLineIndex.sort(key = lambda x : x[0])
      rospy.loginfo("calculate_stop_line_index " + str(self.stopLineIndex))
      pass

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        #calculate once all index of all traffic lights
        self.calculate_stopline_index()
        rospy.loginfo("tl_detector waypoints_cb")

    def traffic_cb(self, msg):
        self.lights = msg.lights
        #rospy.logwarn("TrafficLights update {0}".format(len(self.lights))+ str(self.lights[0]))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        light_wp = self.last_wp
        state = self.state
        self.has_image = True
        self.camera_image = msg
        
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            colorVals = ("Red", "Yellow", "Green", "Unspecified", "Unknown")
            rospy.loginfo("tl_detector detects light change from {0} to {1}"
                          .format(colorVals[self.state], colorVals[state]))
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        wpclosestDist = sys.maxint
        for index in range(len(self.waypoints.waypoints)):
          wp = self.waypoints.waypoints[index]
          wpdist = self.calcDistance_PoseStamped(pose, wp.pose)
          if(wpclosestDist > wpdist):
            wpclosestDist = wpdist
            wpindex = index
        #TODO implement
        return wpindex

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

#         cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #RGB8 in CV2 is not RGB in scipy!!!
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    #get an index in the self.stopLineIndex array which 
    #identified the next stoplight infront of the vehicle
    #Please note: function returns the tl index the car is
    #currently on or it is infront of the car
    #PLEASTE NOTE: self.stopLineIndex is already sorted by waypoint-index
    #TODO: optimize using binary search
    def find_next_stoplineIndex(self, index):
      for i in range (len(self.stopLineIndex)):
        next = (i+1) % len(self.stopLineIndex)
        if self.stopLineIndex[i][0] < index:
          if(self.stopLineIndex[next][0] >= index):
             return next
      return 0

    def get_waypoint_distance(self, idx0, idx1):
      wp0 = self.waypoints.waypoints[idx0]
      wp1 = self.waypoints.waypoints[idx1]
      return self.calcDistance_PoseStamped(wp0.pose, wp1.pose)
    
    
    
    def get_roll_pitch_yaw(self, quaternion):
      return tf.transformations.euler_from_quaternion( [quaternion.x, quaternion.y, quaternion.z, quaternion.w] )
    
    def get_light_status(self, tl_index):
      if not self.has_image:
        return self.lights[tl_index].state
      else:
        # calculate vector from vehicle to traffic light in vehicle coordinate system
        dx_world = self.lights[tl_index].pose.pose.position.x - self.pose.pose.position.x
        dy_world = self.lights[tl_index].pose.pose.position.y - self.pose.pose.position.y
        dz_world = self.lights[tl_index].pose.pose.position.z - self.pose.pose.position.z
        (roll, pitch, yaw) = self.get_roll_pitch_yaw(self.pose.pose.orientation)
        s_y = math.sin(yaw)
        c_y = math.cos(yaw)
        s_p = math.sin(pitch)
        c_p = math.cos(pitch)
        s_r = math.sin(roll)
        c_r = math.cos(roll)
        rotation_matrix = \
            [[c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r], \
            [ s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r], \
            [    -s_p,     c_p*s_r,               c_p*c_r]]
        if np.linalg.matrix_rank(rotation_matrix) == 3:
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            dxyz_vehicle = np.matmul(inv_rotation_matrix, [[dx_world], [dy_world], [dz_world]])
            dy_veh_scaled = dxyz_vehicle[1] / dxyz_vehicle[0]
            dz_veh_scaled = dxyz_vehicle[2] / dxyz_vehicle[0]
            cropped_edge_len = int(round(8000.0 / dxyz_vehicle[0]))
            cropped_x_center = int(round(-2644.0 * dy_veh_scaled + 366.4))
            cropped_y_center = int(round(-2137.0 * dz_veh_scaled + 613.9))
            cropped_x_from = cropped_x_center - (cropped_edge_len/2)
            cropped_y_from = cropped_y_center - (cropped_edge_len/2)
            cropped_x_to   = cropped_x_from   + cropped_edge_len
            cropped_y_to   = cropped_y_from   + cropped_edge_len
            if ( (cropped_x_to - cropped_x_from >= 32) and
                 (cropped_x_from >= 0) and (cropped_x_to < self.camera_image.width) and
                 (cropped_y_from >= 0) and (cropped_y_to < self.camera_image.height) ):
                light_idx = tl_index
                light_pos = self.lights[tl_index].pose.pose.position
                # convert image to cv2 format
                #CV2-RGB8 is the same as the color-scheme understanding of scipy
                #DO NOT APPLY FURTHER CONVERTION
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
                cv_image = cv_image[cropped_y_from:cropped_y_to, cropped_x_from:cropped_x_to]
                result = self.light_classifier.get_classification(cv_image) 
                if (result != self.lights[tl_index].state):
                  rospy.logwarn("Misdetection expected {0} got {1}".format(self.lights[tl_index].state, result))
                  colorVal = ['red', 'yellow', 'green']
                  filename = "./misclassified/mismatch_{0}{1}.jpg".format(colorVal[self.lights[tl_index].state], self.counter)
                  self.counter+=1
                  cv2.imwrite(filename, cv_image)
                return result

        self.has_image = False
        
        return TrafficLight.UNKNOWN

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        if None is self.waypoints:
          return -1, TrafficLight.UNKNOWN
#         if None is self.camera_image:
#           return -1, TrafficLight.UNKNOWN 
        if None is self.pose:
          return -1, TrafficLight.UNKNOWN
        if len( self.stopLineIndex) == 0:
          return -1, TrafficLight.UNKNOWN
        
        car_wp_idx = self.get_closest_waypoint(self.pose)
        closestStopLineIdx = self.find_next_stoplineIndex(car_wp_idx)
        totalNoTL = len(self.stopLineIndex)
        totalNoWP = len(self.waypoints.waypoints)
        
        
        
        #we assume that we can see traffic lights up to a distance of 30 m
        distance = 0.
        #resulting list of stoplights in sight
        tl_index_in_distance = []
        if car_wp_idx == self.stopLineIndex[closestStopLineIdx][0]:
          tl_index_in_distance.append(closestStopLineIdx)
          closestStopLineIdx = (closestStopLineIdx + 1) % len(self.stopLineIndex)
        
        noWPScanned = 0  
        while distance < 40.:
          next_car_wp_idx = (car_wp_idx + 1) % totalNoWP
          distance += self.get_waypoint_distance(car_wp_idx, next_car_wp_idx)
          car_wp_idx = next_car_wp_idx
          #if there is another stoplight in distance, append it to the array 
          #of stoplight indices
          if car_wp_idx == self.stopLineIndex[closestStopLineIdx][0]:
            tl_index_in_distance.append(closestStopLineIdx)
            closestStopLineIdx = (closestStopLineIdx + 1) % len(self.stopLineIndex)
          noWPScanned += 1
#         if(len(tl_index_in_distance ) != 0):
#           rospy.loginfo("In distance of {0:.3f} a total of {1} stoplights identified".format(distance, len(tl_index_in_distance)))
        for idx in tl_index_in_distance:
          trafficLightIndex = self.stopLineIndex[idx][1]
          trafficLightWaypoint = self.stopLineIndex[idx][0]
          if TrafficLight.RED == self.get_light_status(trafficLightIndex):
            return trafficLightWaypoint, TrafficLight.RED
        return -1, TrafficLight.UNKNOWN
              
          
      
#         for idx in range (totalNoTL):
#           closestStopLineIdx = closestStopLineIdx % totalNoTL
#           if TrafficLight.RED == self.lights[self.stopLineIndex[closestStopLineIdx][1]].state:
#             closestRedLightWPIdx = self.stopLineIndex[closestStopLineIdx][0]
#             return closestRedLightWPIdx, TrafficLight.RED
#         # List of positions that correspond to the line to stop in front of for a given intersection
#         stop_line_positions = self.config['stop_line_positions']
#         if(self.pose):
#             car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

#         if light:
#             state = self.get_light_state(light)
#             return light_wp, state
#         #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
