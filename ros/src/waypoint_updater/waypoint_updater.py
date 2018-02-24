#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import copy
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.publish_fwp = rospy.Publisher('/final_waypoints', Lane, queue_size = 10)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.allWaypoints = None #all available waypoints
        self.resultingWaypoints = None
        self.finalWaypoints = None #resulting final waypoints
        self.latestWaypoint = 0 #the last first waypoint
        self.currentPosition = None #the current position
        self.c_maxDistance = None
        self.reducedVelocity = []
        self.obstacleWPIdx = 0. #for trafficlights and obstacles
        
     
        
        rospy.spin()

    def pose_cb(self, msg):
      self.currentPosition = msg;

      #if we aren't yet initialized, we should exit
      if self.allWaypoints is None:
        return      
      foundClosest = False
      wpindex = self.latestWaypoint
      wpprevindex = self.latestWaypoint
      resultingLatestWP = self.latestWaypoint
      wpclosestDist = self.c_maxDistance
      while not foundClosest:
        wpindex = wpindex % len(self.allWaypoints.waypoints)
        wp = self.allWaypoints.waypoints[wpindex].pose
        wpdist = self.distancePos(self.currentPosition.pose, wp.pose)
        if(wpclosestDist > wpdist):
          wpclosestDist = wpdist
        else:
          ##distance is getting larger again
          d1 = wpclosestDist
          d2 = wpdist
          d3 = self.distancePos(self.allWaypoints.waypoints[wpindex].pose.pose,
                                self.allWaypoints.waypoints[wpprevindex].pose.pose)
          if(d2 > d3):
            resultingLatestWP = wpprevindex
          else:
            resultingLatestWP = wpindex
          foundClosest = True
        wpprevindex = wpindex
        wpindex += 1
      
      if(self.latestWaypoint != resultingLatestWP):
        rospy.logwarn("Clostest waypoint has changed from {0} => {1} at location {2}"
                      .format(self.latestWaypoint, resultingLatestWP, str(self.currentPosition.pose)))
        self.latestWaypoint = resultingLatestWP
        #identify the final 200 waypoints and publish them
        self.filter_and_send_waypoints()
      
      
      pass

    def waypoints_cb(self, waypoints):
      self.allWaypoints = waypoints
      #get the possible max distance (from start to middle of track)
      #=> the track is a circle
      self.c_maxDistance = self.distancePos(self.allWaypoints.waypoints[0].pose.pose, 
                           self.allWaypoints.waypoints[len(self.allWaypoints.waypoints)/2].pose.pose)
      self.resultingWaypoints = copy.deepcopy(self.allWaypoints.waypoints)
      rospy.logwarn("WAYPONIT_CB")
      pass
      
    
    def filter_and_send_waypoints(self):
      rWaypoints = Lane()  #copy the list
      pos = self.latestWaypoint
      wp = self.allWaypoints.waypoints
      rWaypoints.header = self.allWaypoints.header
      rWaypoints.waypoints = wp[pos:min(pos+LOOKAHEAD_WPS, len(wp))]
      size = len(rWaypoints.waypoints)
      if size < LOOKAHEAD_WPS:
        rospy.logwarn("\n\n\n\nRoundabout")
        rWaypoints.waypoints.append(wp[:LOOKAHEAD_WPS-size])

      speedArray = []
      for i in range (10):
        speedArray.append([pos+i, self.get_waypoint_velocity(rWaypoints.waypoints[i])])
      rospy.logwarn("Next guys "+str(speedArray))
      self.publish_fwp.publish(rWaypoints)
      pass
        
      
    def same_waypoints(self, wp1, wp2):
      return wp1.pose.position == wp2.pose.position;

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        resultingWP = self.allWaypoints.waypoints
        
        
        waypoint = int(msg.data)
        self.obstacleWPIdx = waypoint
        deacceleration = 10.
        securityDistance = 6. #size of the crossing
        currentSpeed = self.get_waypoint_velocity(self.allWaypoints.waypoints[self.latestWaypoint])
        if(waypoint != -1):
          #calculate distance between car and light
          distCar2TL = self.distance(self.allWaypoints.waypoints, 
                                     self.latestWaypoint, self.obstacleWPIdx)
          timeToZeroSpeed = currentSpeed / deacceleration
          distanceToStop =securityDistance + 0.5 * deacceleration * timeToZeroSpeed**2; 
          rospy.logwarn("No problem distance is {0} and it will take us {1}".format(distCar2TL, distanceToStop))
          
          
          speed = 0.
          prevPose = resultingWP[waypoint].pose.pose
          distance = 0.
          iterator = 0
          while distance < securityDistance:
            iterator +=1
            self.reducedVelocity.append([waypoint, resultingWP[waypoint].twist.twist.linear.x])
            resultingWP[waypoint].twist.twist.linear.x = speed
            prevWaypoint = waypoint
            waypoint = (waypoint - 1) % len(self.allWaypoints.waypoints)
            distance += self.distance(self.allWaypoints.waypoints, waypoint, prevWaypoint)
          waypoint = (waypoint + 1) % len(self.allWaypoints.waypoints)
          rospy.logwarn("Total of {0} wp in security distance".format(iterator))
          stopLoop = False
          iterator = 0
          while not stopLoop:
            waypoint = (waypoint - 1) % len(self.allWaypoints.waypoints)
            distance = self.distancePos(resultingWP[waypoint].pose.pose, prevPose)
            time = math.sqrt(distance * 2 / deacceleration)
            rospy.logwarn("Timer " + str(time) + " Dist " +str(self.distancePos(resultingWP[waypoint].pose.pose, prevPose)))
            prevPose = resultingWP[waypoint].pose.pose
            speed = speed + distance / time
            #rospy.logwarn("Altered wp {0}".format(waypoint))
            if(resultingWP[waypoint].twist.twist.linear.x < speed):
              stopLoop = True
            else :
              self.reducedVelocity.append([waypoint, resultingWP[waypoint].twist.twist.linear.x])
              resultingWP[waypoint].twist.twist.linear.x = speed
#               rospy.logwarn(str(resultingWP[waypoint]))
              rospy.logwarn("Altered wp {0} - speed {1}".format(waypoint, speed))
            iterator += 1
          rospy.loginfo("Altered a total of {0} waypoints".format(iterator))
        elif 0 != len(self.reducedVelocity):
          for entry in self.reducedVelocity:
            resultingWP[entry[0]].twist.twist.linear.x = entry[1]
          self.reducedVelocity = [] 
#         
        self.filter_and_send_waypoints()
           
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def getWaypointsFrom(self, originalLane, startIndex):
      result = Lane()
      owp = originalLane.waypoints
      result.waypoints = owp[startIndex:min(len(owp), startIndex+LOOKAHEAD_WPS)]
      result.waypoints.append(owp[:(LOOKAHEAD_WPS - len(result.waypoints))])
      return result
    #expects PoseStamped/Pose as argument
    def distancePos(self, pos1, pos2):
      return math.sqrt((pos1.position.x-pos2.position.x)**2 + (pos1.position.y-pos2.position.y)**2)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
