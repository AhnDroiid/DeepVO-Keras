import math

DEG2RAD = math.pi / 180.0
class Vehicle:
    def __init__(self, steering_ratio = 15.56, x=0, y=0, velocity = 0, steering = 0):
        self.steering_ratio = steering_ratio #steering / yaw  #14.60 good
        self.x = x
        self.y = y
        self.velocity = velocity
        self.steering = steering # radian
        self.yaw = self.steering/self.steering_ratio
        self.wheel_base = 2.845


    def BicycleModel(self, current_steering, current_velocity, dt, forward, alpha = 0.5):
        current_velocity *= 1.2
        current_velocity = alpha*current_velocity+(1-alpha)*self.velocity  # control
        if forward:
            sign = +1.
        else:
            sign = -1.
        
        current_yaw = DEG2RAD * current_steering/self.steering_ratio # theta(t)
        slip_beta = math.atan2(math.tan(current_yaw), 2)
        self.x += (sign) * current_velocity * math.cos(self.yaw + slip_beta)*dt
        self.y -= (sign) * current_velocity * math.sin(self.yaw + slip_beta)*dt
        self.yaw += dt * (sign) * current_velocity * math.tan(current_yaw) * math.cos(slip_beta) / self.wheel_base
        self.velocity = current_velocity

