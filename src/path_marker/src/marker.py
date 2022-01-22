#! /usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TwistStamped, Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA, String
from nav_msgs.msg import Odometry

class TrajectoryInteractiveMarkers:

    def __init__(self):
        self.count = 0 
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=5)
        rospy.Subscriber('odom', Odometry, self.event_in_cb)
        rospy.sleep(0.5)

    def event_in_cb(self,msg):
        self.odometry = msg
        self.a = list()
        self.a.append(self.odometry.pose.pose.position.x)
        self.a.append(self.odometry.pose.pose.position.y)
        self.a.append(self.odometry.pose.pose.position.z)
        self.show_text_in_rviz()

    def show_text_in_rviz(self):
        self.marker = Marker()
        self.marker = Marker(
            type=Marker.CYLINDER,
            id=0,
            lifetime=rospy.Duration(500),
            pose=Pose(Point(self.a[0]/10**5,self.a[1]/10**5,self.a[2]/10**5), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.02, 0.02, 0.02),
            header=Header(frame_id='base_link'),
            color=ColorRGBA(0.0, 2.0, 0.0, 0.8))
        self.count+=1
        self.marker.id = self.count
        self.marker_publisher.publish(self.marker)
        #rospy.loginfo('msg published')

if __name__ == '__main__':
    rospy.init_node("trajectory_interactive_markers_node", anonymous=True)
    trajectory_interactive_markers = TrajectoryInteractiveMarkers()
    rospy.sleep(0.5)
    rospy.spin()