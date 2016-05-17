'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random
import csv
import origdriverx
import copy
import math
import collections


from sklearn.utils.extmath import cartesian

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, args):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = args.stage
        self.iamironman = True
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.arguments = args
        self.dq = collections.deque(maxlen=200)



        self.steer_lock = 0.785398
        self.max_speed = 86
        self.max_speed_distance = 70
        self.steer_sensitivity_offset = 70
        # parameters

        self.exp_rate = 0.1 # exploration rate
        self.mu = 0.4
        self.alpha = 0.0005 # learning rate
        self.gamma = 0.99
        self.lmbd = 0.8
        self.degrade_mu_in = 30
        self.dec_rate = self.mu / self.degrade_mu_in
        if not self.arguments.enable_exploration:
            self.exp_rate = 0
            self.mu = 0.0

        self.mem = collections.deque(maxlen=700)
        self.nz = collections.deque(maxlen=500)

        if self.arguments.log_level=='DEBUG':
            self.debug = True
        else:
            self.debug = False
            
        # TODO should be -11 first one isntead of -15
        self.feature_tiles = {#'speedx':[[-15,  0, 50, 90, 140, 300],
                'trackpos':[-2, -0.6, -0.2, 0.2, 0.6,  2],
                'angle':[-2, -0.8, -0.3, 0, 0.4, 2],
                #'track':np.array([-1, 10, 29, 50, 100, 150, 201]) / 200.}
                }


        tiles = {'trackpos' : [[-2, -0.8, -0.3, 0, 0.4, 2],
            [-2, -0.7, -0.2, 0.05, 0.5, 2],
            [-2, -0.6, -0.15, 0.1, 0.6, 2],
            [-2, -0.5, -0.1, 0.2, 0.7, 2],
            [-2, -0.4, -0.05,  0.3, 0.9, 2]],
            'angle': [[-2, -0.8, -0.3, 0, 0.4, 2],
                [-2, -0.7, -0.2, 0.05, 0.5, 2],
                [-2, -0.6, -0.15, 0.1, 0.6, 2],
                [-2, -0.5, -0.1, 0.2, 0.7, 2],
                [-2, -0.4, -0.05,  0.3, 0.9, 2]],
            'speedx':[[-15, 0, 40, 100, 120]]
            }

        for nm, tls in tiles.items():
            self.feature_tiles[nm] = []
            for tl in tls:
                for j in range(len(tl) - 1):
                    self.feature_tiles[nm].append((tl[j], tl[j + 1]))

        print self.feature_tiles

        self.feature_tilep = {#'speedx':[[-15,  0, 50, 90, 140, 300],
                'trackpos':[[-2, -0.4],[-0.6, 0.1], [-0.2, 0.3], [0.1, 0.6], [0.4, 2]],
                'angle':[-2, -0.8, -0.3, 0, 0.4, 2],
                #'track':np.array([-1, 10, 29, 50, 100, 150, 201]) / 200.}
                }
        
        self.feature_tilep = {'speedx':[-15, 0, 10, 30, 50, 80, 100, 120, 150, 200],
                'trackpos':[-2, -0.6, -0.2, 0.2, 0.6, 2],
                #'trackpos':[-2, -1, -0.6,-0.4 -0.2, 0, 0.2, 0.4, 0.6, 1, 2],
                #'angle':[-2, -0.8, -0.3, 0, 0.4, 2],
                #'track':np.array([-1, 10, 29, 50, 100, 150, 201]) / 200.}
                }

        #self.action_map = {'steer': [0.5, 0.1, 0, -0.1, -0.5], 'accel':[0, 1]}
        self.action_map = {'steer': [ 0.5, 0.05, 0, -0.05, -0.5], 'accel':[1]}
        

        # create feature dim
        pp = sorted(self.feature_tiles.items()) 
       
        dims_feature = []
        for nm, xs in pp:
            if nm == 'track':
                for _ in range(5):
                    dims_feature.append(len(xs) - 1)
            else:
                dims_feature.append(len(xs))
        
        
        print dims_feature

        dims_action = [len(x) for x in self.action_map.values()]
       
        self.action_count = 1
        for x in dims_action:
            self.action_count *= x
    
        dims_feature.append(self.action_count)
        self.w = np.zeros(tuple(dims_feature))
        print self.w.shape
        if self.arguments.load_weights:
            self.w = np.load(self.arguments.load_weights)


        # set old state
        self.s = None
        self.et = np.zeros_like(self.w) # eligibility trace
        
        
        self.total_train_steps = 0
        self.skip = args.skip

        self.prev_rpm = None
        self.episode = 0
        self.distances = []
        self.onRestart()

    def speedsteer_to_act(self, speed, steer):
        steers = np.abs(steer - np.array(self.action_map['steer']))
        accel = np.abs(speed - np.array(self.action_map['accel']))
        st = np.argmin(steers)
        accel = np.argmin(accel)
        cta = len(self.action_map['accel'])

        return st * cta + accel

    def act_to_speedsteer(self, act):
        #print 'acttospeedsteer inp', act
        ttc = len(self.action_map['accel'])
        steer = self.action_map['steer'][act / ttc]
        speed = self.action_map['accel'][act % ttc]
        #print 'acttospeedsteer outp', speed ,steer
        return speed, steer


    def stateact_to_feature(self, state, act, onlyindex=True):
        zedaind = []
        for nm, xs in sorted(self.feature_tiles.items()):
            val = None
            if nm == 'speedx':
                val = state.getSpeedX()
            elif nm == 'trackpos':
                val = state.getTrackPos()
            elif nm == 'angle':
                val = state.getAngle()
            
            #print val, nm
            inds = []
            if not val == None:
                # on of the above
                for i in range(len(xs) - 1):
                    if xs[i][0] <= val < xs[i + 1][1]:
                        inds.append(i)

                zedaind.append(inds)

            elif nm == 'track':
                # remaning are trackpositions, lets get them
                tracks = np.array(state.getTrack()) / 200.
                sensors = []

                sensors.append(tracks[3]) # -40
                sensors.append((tracks[4] + tracks[5] + tracks[6])/3.)
                sensors.append((tracks[9] + tracks[8] + tracks[10]) / 3.) # 0
                sensors.append((tracks[12] + tracks[13] + tracks[14])/3.)
                sensors.append(tracks[15])
                if self.arguments.show_sensors:
                    print sensors
                for val in sensors:
                    for i in range(len(xs) - 1):
                        if xs[i] <= val <= xs[i + 1]:
                            ind.append(i)
                            break
            else:
                assert False
        zedaind.append([act]) 
        #print 'feature shape-', self.w.shape,'index length-',  len(ind)
        #print ind
        assert len(zedaind) == len(self.w.shape), 'ind %s, w %s' %(str(ind), str(self.w.shape))
        if onlyindex:
            return tuple(ind)
        else:
            ft = np.zeros_like(self.w)

            for tot in cartesian(zedaind):
                ft[tuple(tot)] = 1
            return ft
   

    def get_val(self, w, state, act):
        features = self.stateact_to_feature(state, act, onlyindex=False)
        #print features, act
        #print w.shape
        #print 'act_coun', self.action_count
        #return w[features]
        return np.sum(w * features)
        
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [-90, -75, -60, -40, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 60, 75, 90]
        return self.parser.stringify({'init': self.angles})
    
    """
    def getState(self):
        #state = np.array([self.state.getSpeedX() / 200.0, self.state.getAngle(), self.state.getTrackPos()])
        state = np.array(self.state.getTrack() + [self.state.getSpeedX()]) / 200.0
        state = np.array(self.state.getTrack()) / 200.0
        assert state.shape == (self.num_inputs,)
        return state
    """

    def getReward(self, terminal):
        if terminal:
            reward = -4000
        else:
            dist = self.state.getDistFromStart()
            fulldist = self.state.getDistRaced()

            prev_dist = self.s.getDistFromStart()
            prev_fulldist = self.s.getDistRaced()
            #print 'prev', 'cur', prev_dist, dist
            if prev_dist is not None:
                #reward = max(0, dist - prev_dist) * 10
                #assert reward >= 0, "reward: %f" % reward
                reward = (dist - prev_dist) * 10
            else:
                reward = 0
                
                # try to stay center 
            centerReward = max(0, (0.5 - np.abs(self.state.getTrackPos())) * 100)
            if self.state.getSpeedX() <40:
                centerReward = 0 
            #print 'centerRew', centerReward
            reward += centerReward

            # try to not be at the same place
            #if self.state.getSpeedX() < 10:
                #reward -= (10 - self.state.getSpeedX())

        return reward
            

    def getTerminal(self):
        #print self.state.getTrackPos()
        #print 'angle', self.state.getAngle()
        threshold = 0.96
        angle_thr = 2

        self.dq.append(self.state.getSpeedX())
        if len(self.dq) == self.dq.maxlen and np.average(self.dq) < 8:
            return True
        
        if np.abs(self.state.getAngle()) > angle_thr:
            return True
        if self.state.getTrackPos() > threshold or self.state.getTrackPos() < -threshold:
            return True
        return np.all(np.array(self.state.getTrack()) == -1)

    def drive(self, msg):
        # parse incoming message
        self.state.setFromMsg(msg)
        
        # show sensors
        #if self.show_sensors:
            #self.stats.update(self.state)

        # training
        """
        if self.enable_training and self.mem.count >= self.minibatch_size:
          minibatch = self.mem.getMinibatch()
          self.net.train(minibatch)
          self.total_train_steps += 1
          #print "total_train_steps:", self.total_train_steps
        """

        # skip frame and use the same action as previously
        if self.skip > 0:
            self.frame = (self.frame + 1) % self.skip
            if self.frame != 0:
                return self.control.toMsg()

        # fetch state, calculate reward and terminal indicator  
        #state = self.getState()
        if self.s == None:
            self.s = copy.deepcopy(self.state)
            self.a = 2
            return # first state during beginning nothing to do

        #ft = self.stateact_to_featue(self.state, 10)
        #self.s = self.state
        terminal = self.getTerminal()
        reward = self.getReward(terminal)
        if terminal:
            pass
            #assert False
        
        s_prime = copy.deepcopy(self.state)
        s = self.s
        a = self.a 
        r = reward
        self.mem.append((s, a, r))
        
        if not terminal:
            a_prime = self.e_greedy(s_prime, self.w)
            #nextq = self.get_val(self.w, s_prime, a_prime)
        else:
            a_prime, nextq = 0, 0 # not important

        self.a = a_prime
        self.s = s_prime
        #print "reward:", reward

        # if terminal state (out of track), then restart game
        if terminal and self.arguments.enable_exploration:
            #print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)
            

         
        #gear = origdriverx.Driver.gear(self)
        gear = self.gear()
        # set actions
        speed, steer = self.act_to_speedsteer(a_prime)
        #self.setSteerAction(steer)
        #self.setGearAction(gear)
        #self.setSpeedAction(speed)
        #speed = self.steerx()
        self.control.setSteer(steer)
        self.control.setGear(gear)

        if speed > 0:
            self.control.setAccel(speed)
            self.control.setBrake(0)
        else:
            self.control.setAccel(0)
            self.control.setBrake(-speed)
        
        if self.state.getSpeedX() > self.max_speed:
            self.control.setBrake(0.1)
            self.control.setAccel(0)
        
        # remember state and actions 
        #print "total_train_steps:", self.total_train_steps, "mem_count:", self.mem.count

        #print "reward:", reward, "epsilon:", epsilon

        return self.control.toMsg()

    
    def speedx(self):

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
        #self.control.setAccel(ret)
        return ret
        #return ret

    def e_greedy(self, state, w):
        flip = np.random.random()
        if flip <= self.exp_rate:
            # do random 
            ret = np.random.choice(self.action_count)
            if self.debug:
                pass
                #print 'random_choiced act', ret
            return ret
        elif self.exp_rate < flip <= self.exp_rate + self.mu:
            steer = self.steer()
            accel = self.speedx()
            print 'suggested steer, accel', steer, accel
            if self.debug:
                print 'suggested steer, accel', steer, accel
                print 'returned ', self.speedsteer_to_act(accel, steer) 
            #print 'steer is', steer, 'accel is', accel
            #print 'a_prime is', self.speedsteer_to_act(accel, steer)
            return self.speedsteer_to_act(accel, steer)
        else:
            # take argmax
            xs = []
            for act in range(self.action_count):
                xs.append(self.get_val(w, self.state, act))
            #ret = rargmax(xs)
            ret = np.argmax(xs)
            if self.debug:
                print 'xs is', xs
                print 'argsorted choice', np.argsort(xs)[::-1]
                print 'returned', ret
            self.nz.append(len(np.nonzero(xs)[0]))
            return ret


    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        ret = (angle - dist*0.5)/self.steer_lock
        return ret


    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            #accel += 0.1
            accel = 1
            if accel > 1:
                accel = 1.0
        else:
            #accel -= 0.1
            accel = 0
            if accel < 0:
                accel = 0.0
        return accel   


    def steerx(self):
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
    
    def gearx(self):
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
        """
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
        """
        return gear


    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
       
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000 and gear < 6:
            gear += 1
        
        if not up and rpm < 3000 and gear > 0:
            gear -= 1
        
        return gear
        

    def setSpeedAction(self, speed):
        #assert 0 <= speed <= self.num_speeds
        accel = self.speeds[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)
    
    def onShutDown(self):
        pass

    def onRestart(self):

        self.s = None # set current state to None
        self.dq.clear()
        self.mu = max(0, self.mu - self.dec_rate)
        
        ###### begin TRAIN
        #print 'mgelo', len(self.mem)
        
        self.et = np.zeros_like(self.w)
        self.dwsum = np.zeros_like(self.w)
        self.mem.append((1, 2, -500000)) #dummy terminal
        #st_point = max(0, len(self.mem) - 1000)
        #print st_point, len(self.mem)
        if self.arguments.enable_exploration:
            for i in xrange(len(self.mem)):
                s, a, r = self.mem[i]
                #print 'act, rew' , a, r
                terminal  = (i == len(self.mem) - 1)
                if terminal:
                    break

                s_prime, a_prime, rr = self.mem[i + 1]
                if not (i + 1 == len(self.mem) - 1):
                    #a_prime = self.e_greedy(s_prime, self.w)
                    nextq = self.get_val(self.w, s_prime, a_prime)
                else:
                    a_prime, nextq = 0, -5000 # not important

                #print 'rew', r
                curq = self.get_val(self.w, s, a)
                delta = r + self.gamma * nextq - curq

                #ind_of_stateact = self.stateact_to_feature(s, a)
                stateact = self.stateact_to_feature(s, a, onlyindex=False)
                self.et = self.gamma * self.lmbd * self.et
                self.et += stateact

                dew = self.alpha * delta * self.et
                self.dwsum += dew
            self.w += self.dwsum
                #print np.nonzero(dew)
                #rprint self.w


            ####  END TRAIN

            # save weights
            np.save('temp_'+str(self.episode), self.w)

        
        self.mem.clear() # clear memory
    
        if self.episode > 0:
            dist = self.state.getDistRaced()
            self.distances.append(dist)
            print "Episode:", self.episode, "\tDistance:", dist, "\tMax:", max(self.distances), "\tMedian10:", np.median(self.distances[-10:]), "\tavg nonzero", np.average(self.nz)

        self.nz.clear()
        self.episode += 1

