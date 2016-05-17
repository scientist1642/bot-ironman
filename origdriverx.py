# improved origdriver according to paper

'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import math

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 120
        self.max_speed_distance = 70
        self.steer_sensitivity_offset = 80
        self.prev_rpm = None
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5

        print self.angles
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        self.steer()
        
        self.gear()
        
        self.speed()
        
        return self.control.toMsg()
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        target_angle = (angle - dist * 0.5)
        current_speed = self.state.getSpeedX()
        ret1 = target_angle / (self.steer_lock * (current_speed - self.steer_sensitivity_offset))
        ret2 = target_angle / self.steer_lock

        if current_speed > self.steer_sensitivity_offset:
            ret = ret1
        else:
            ret = ret2

        #self.control.setSteer((angle - dist*0.5)/self.steer_lock)
        #print ret1, ret2

        self.control.setSteer(ret2)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if gear > 2 and self.state.getSpeedX() <= 60:
            # recover after crashing
            gear = 2
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        self.prev_rpm = rpm
      
        ups = [8000, 9500, 9500,9500, 9500, 0]
        downs = [0, 4000, 6300, 7000, 7300, 7300]
        if up:
            now = gear - 1
            if rpm > ups[now]:
                gear += 1
        else:
            now = gear - 1
            if rpm < downs[now]:
                gear -= 1
        gear = min(3, gear)
        gear = max(2, gear)
        if not hasattr(self,'iamironman'):
            self.control.setGear(gear)
        """
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
        """
        return gear
    
    def speed(self):

        tracks = self.state.getTrack()
        current_speed = self.state.getSpeedX()

        right_sensor = tracks[10] 
        central_sensor = tracks[9]
        leftSensor = tracks[8]

        if (central_sensor > self.max_speed_distance) or (central_sensor >= right_sensor and central_sensor >= leftSensor):
                target_speed = self.max_speed
        else:
            if right_sensor > leftSensor:
                h = central_sensor * math.sin(math.radians(5))
                b = right_sensor - central_sensor * math.cos(math.radians(5))
                sin_angle = b * b / (h * h + b * b)
                target_speed =self.max_speed * (central_sensor * sin_angle / self.max_speed_distance)
            else:
                h = central_sensor * math.sin(math.radians(5))
                b = leftSensor - central_sensor * math.cos(math.radians(5))
                sin_angle = b * b / (h * h + b * b)
                target_speed = self.max_speed * (central_sensor * sin_angle / self.max_speed_distance)
        
        ret = 2 / (1 +  math.exp(current_speed - target_speed)) - 1
       # print ret
        self.control.setAccel(ret)
        #return ret

        """
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
        """
            
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        
