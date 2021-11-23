import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
from . import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.
        us = np.random.multivariate_normal(u,self.R,self.M)
        self.xs = self.transition_model(us, dt)

        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.

#        i=0
#        c = ws[0]
#        print("xs.shape, ws.shape",xs.shape, ws.shape)
#        print("xs",xs)
#        print("ws",ws)
        
        u_test = np.sum(ws)*(r+np.linspace(0,self.M-1,num=100)/self.M)
#        print("u_test",u_test)

        ws_test = ws.cumsum(axis=0)
#        print("ws_test",ws_test)

        index = np.searchsorted(ws_test, u_test)
#        print("index",index,index.shape)

        self.xs = xs[index]
        self.ws = ws[index]

#        for m in range(self.M):
#            print("ws[m]",ws[m])
#            u = np.sum(ws)*(r+m/self.M)
#            while c < u:
#                i = i+1
#                c = c+ws[i]

#            print("i",i)
#            self.xs[m] = xs[i]
#            self.ws[m] = ws[i]

        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them

#        print("us.shape, us",us.shape, us)
#        print("self.xs.shape, self.M",self.xs.shape,self.M)


        x_t_1,y_t_1,th_t_1 = self.xs.T
#        print("x_t_1,y_t_1,th_t_1",x_t_1.shape,y_t_1.shape,th_t_1.shape)

        v_t, om_t = us.T
#        print("v_t, om_t",v_t.shape, om_t.shape)

        index1 = np.where(abs(om_t)<tb.EPSILON_OMEGA)
        index2 = np.where(abs(om_t)>=tb.EPSILON_OMEGA)

 #       print("index1.shape,index2.shape",len(index1[0]),len(index2[0]),index1,index2)

        x_t = np.zeros_like(x_t_1)
        y_t = np.zeros_like(y_t_1)
        th_t = th_t_1 + om_t*dt
#        print("th_t.shape",th_t.shape)

        x_t[index1] = x_t_1[index1] + v_t[index1]*np.cos(th_t_1[index1])*dt
        y_t[index1] = y_t_1[index1] + v_t[index1]*np.sin(th_t_1[index1])*dt

#        print("x_t.shape,y_t.shape",x_t.shape,y_t.shape)

        x_t[index2] = x_t_1[index2] + v_t[index2]/om_t[index2]*(np.sin(th_t[index2])-np.sin(th_t_1[index2]))
        y_t[index2] = y_t_1[index2] - v_t[index2]/om_t[index2]*(np.cos(th_t[index2])-np.cos(th_t_1[index2]))

#        print("after index2")
        g = np.vstack((x_t,y_t,th_t)).T
#        print("g_test.shape, g_test",g_test.shape, g_test)

#    x_t_1 = xvec[0]
#    y_t_1 = xvec[1]
#    th_t_1 = xvec[2]

#    v_t = u[0]
#    om_t = u[1]

#    th_t = th_t_1 + om_t*dt

#    if abs(om_t)<EPSILON_OMEGA:
#        x_t = x_t_1 + v_t*np.cos(th_t_1)*dt
#        y_t = y_t_1 + v_t*np.sin(th_t_1)*dt

#        g = np.array([x_t,y_t,th_t])


#    else:
 #       x_t = x_t_1 + v_t/om_t*(np.sin(th_t)-np.sin(th_t_1))
#        y_t = y_t_1 - v_t/om_t*(np.cos(th_t)-np.cos(th_t_1))

#        g = np.array([x_t,y_t,th_t])
    



#        g=np.zeros((self.M,3))

#        for i in range(self.M):
#            g[i,:] = tb.compute_dynamics(self.xs[i,:], us[i], dt, compute_jacobians=False)

#        print("g.shape, g",g.shape, g)
        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        vs, Q = self.measurement_model(z_raw, np.asarray(Q_raw))
        if vs is None:
            return

        ws = scipy.stats.multivariate_normal.pdf(vs,cov=Q)

        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.asarray(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful

 #       vs = vs.reshape(-1,1).flatten()
 #       print("z.shape,z",z.shape,z)

        Q = scipy.linalg.block_diag(*Q_raw)
#        print("Q.shape,Q",Q.shape,Q)

        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.
        hs = self.compute_predicted_measurements()
        numi=z_raw.shape[1]
        numj=hs.shape[2]
#        vs = np.zeros((self.M,numi,2))
#        V = np.zeros((self.M,numi,numj,2))
#        d = np.zeros((numi,numj))
#        print("numi,numj",numi,numj)
#        print("z_raw.shape,hs.shape,z_raw,hs",z_raw.shape,hs.shape,z_raw,hs)

#        print("z_raw[None,0,:,None]",z_raw[None,0,:,None].shape)
#        print("hs[:,0,:,None,:]",hs[:,0,None,:].shape)
#        print("angle_diff(z_raw[None,0,:,None], hs[:,0,None,:]",angle_diff(z_raw[None,0,:,None], hs[:,0,None,:]).shape)
#        print("z_raw[None,1,:,None] - hs[:,1,None,:]",(z_raw[None,1,:,None] - hs[:,1,None,:]).shape)

        V_test = np.array([angle_diff(z_raw[None,0,:,None], hs[:,0,None,:]),(z_raw[None,1,:,None] - hs[:,1,None,:])]).transpose(1,3,2,0)
#        print("V_test.shape",V_test.shape)

        V_test = V_test[:,:,:,:,None]
#        print("V_test[:,:,:,:,None].shape",V_test.shape)

        Q = np.linalg.inv(Q_raw)[None,None,:,:,:]
#        print("Q.shape",Q.shape)

        d_test = np.matmul(np.matmul(V_test.transpose(0,1,2,4,3),Q),V_test)
#        print("d_test.shape",d_test.shape)
#        print("d_test.reshape(self.M,numj,numi)",d_test.reshape(self.M,numj,numi).shape)

        d_test = d_test.reshape(self.M,numj,numi).transpose(0,2,1)
#        print("d_test.shape",d_test.shape)
        m_index = np.argmin(d_test,axis=2)[:, None, :, None]
#        print("m_index,",m_index.shape,m_index)

#        print("V_test.reshape(self.M,numj,numi,2).transpose(0,2,1,3)",V_test.reshape(self.M,numj,numi,2).transpose(0,2,1,3).shape)
#        vs_test = V_test.reshape(self.M,numj,numi,2).transpose(0,2,1,3)[m_index]
        vs_test = np.take_along_axis(V_test.reshape(self.M,numj,numi,2),m_index,axis=1)
        vs = vs_test.reshape(self.M,numi,2)

#        print("vs_test.shape,vs_test",vs_test.shape,vs_test)

#        for m in range(self.M):
#            for i in range(numi):
#                for j in range(numj):
#                    print("z_raw[0,i], hs[0,j],z_raw[1,i],hs[0,j]",z_raw[0,i], hs[0,j],z_raw[1,i],hs[0,j])
#                    V[m,i,j,0] = angle_diff(z_raw[0,i], hs[m,0,j])
#                    V[m,i,j,1] = z_raw[1,i] - hs[m,1,j]
#                    print("V[m,i,j]",V[m,i,j])

#                    d[i,j]=np.dot(np.dot(V[m,i,j].T,np.linalg.inv(Q_raw[i,:,:])),V[m,i,j])
#                    print("np.dot(V[m,i,j].T,np.linalg.inv(Q_raw[i,:,:]))",np.dot(V[m,i,j].T,np.linalg.inv(Q_raw[i,:,:])).shape)
#                    print(d[i,j])
#                    print("d[i,j]",d[i,j])

#                min_index=np.argmin(d[i,:])
#                print("min_index",min_index)
#                vs[m,i,:] = V[m,i,min_index,:]

#        print("vs.shape,vs",vs.shape,vs)
        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.
#        print("self.map_lines.shape,self.M",self.map_lines.shape,self.M)

        hs = np.zeros((self.M,2,self.map_lines.shape[1]))
        x_base, y_base, th_base = self.tf_base_to_camera

        x_cam,y_cam,th_cam = np.vstack(([self.xs[:,0]+x_base*np.cos(self.xs[:,2])-y_base*np.sin(self.xs[:,2])],[self.xs[:,1] + x_base*np.sin(self.xs[:,2]) + y_base*np.cos(self.xs[:,2])],[self.xs[:,2] + th_base]))
        alpha, r = self.map_lines

        alpha_cam = alpha[None,:] - th_cam[:,None]
        r_cam = r[None,:]-np.dot(x_cam.reshape(self.M,1),np.cos(alpha).reshape(1,self.map_lines.shape[1]))-np.dot(y_cam.reshape(100,1),np.sin(alpha).reshape(1,self.map_lines.shape[1]))

        #normalize_line_parameters
        index = np.where(r_cam < 0)
        alpha_cam[index] += np.pi
        r_cam[index] *= -1
        alpha_cam = (alpha_cam + np.pi) % (2*np.pi) - np.pi

        hs = np.array([alpha_cam,r_cam]).transpose(1,0,2)

#        for m in range(self.M):
#            for j in range(self.map_lines.shape[1]):

#                h,Hx = tb.transform_line_to_scanner_frame(self.map_lines[:,j], self.xs[m,:], self.tf_base_to_camera)
            
#                h, Hx = tb.normalize_line_parameters(h, Hx)
#                hs[m,:,j] = h

#        print("hs",hs)
        ########## Code ends here ##########

        return hs

