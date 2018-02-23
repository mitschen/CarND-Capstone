
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

import rospy
from pid import PID
from yaw_controller import YawController
from geometry_msgs.msg import TwistStamped
from lowpass import LowPassFilter

class Controller(object):
    def __init__(self, yaw_ctrl, acc_pid, brake_factor, timestep):
        # TODO: Implement
        self.yawCtrl = yaw_ctrl
        self.pid = acc_pid
        self.brake_factor = brake_factor
        self.timestep = timestep
        self.lowpass = LowPassFilter(0.1, self.timestep)
        pass

    def control(self,linear_velocity, angular_velocity, current_velocity, enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        error = linear_velocity - current_velocity
        acc = self.lowpass.filt(self.pid.step(error, self.timestep))
        if acc < 0.:
          brake = -acc * self.brake_factor 
          acc = 0.
        else:
          brake = 0.
        if enabled:
          return acc, brake, self.yawCtrl.get_steering(linear_velocity, angular_velocity, current_velocity)
        self.pid.reset()
        return 0., 0., 0.
