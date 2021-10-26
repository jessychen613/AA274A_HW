import numpy as np
from utils import wrapToPi

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        rho = np.sqrt(np.square(self.y_g-y) + np.square(self.x_g-x))
 #       alpha = wrapToPi(np.arctan2(y,x) - th)
 #       delta = wrapToPi(np.arctan2((self.y_g),(self.x_g)) - self.th_g)
        delta = wrapToPi(np.arctan2((self.y_g-y),(self.x_g-x)) - self.th_g)
        alpha = wrapToPi(np.arctan2((self.y_g-y),(self.x_g-x)) - th)

        if (rho<=RHO_THRES) and (alpha<=ALPHA_THRES) and (delta<=DELTA_THRES):
            V = 0
        else:
            V = self.k1 * rho * np.cos(alpha)

#        om = self.k2 * alpha + self.k1*((np.sinc(alpha) * np.cos(alpha)))*(alpha+self.k3*delta)
        om = self.k2 * alpha + self.k1*((np.sin(alpha) * np.cos(alpha))/alpha)*(alpha+self.k3*delta)

        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om