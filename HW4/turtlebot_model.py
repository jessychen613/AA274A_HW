import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

#    print("xvec, u, dt",xvec, u, dt)

    x_t_1 = xvec[0]
    y_t_1 = xvec[1]
    th_t_1 = xvec[2]

    v_t = u[0]
    om_t = u[1]

    th_t = th_t_1 + om_t*dt

    if abs(om_t)<EPSILON_OMEGA:
        x_t = x_t_1 + v_t*np.cos(th_t_1)*dt
        y_t = y_t_1 + v_t*np.sin(th_t_1)*dt

        g = np.array([x_t,y_t,th_t])

        Gx = np.array([[1,0,-np.sin(th_t_1)*v_t*dt],[0.,1.,np.cos(th_t_1)*v_t*dt],[0.,0.,1.]])
        Gu = np.array([[np.cos(th_t_1)*dt,-v_t/2*np.sin(th_t_1)*np.square(dt)],[np.sin(th_t_1)*dt,v_t/2*np.cos(th_t_1)*np.square(dt)],[0.,dt]])

    else:
        x_t = x_t_1 + v_t/om_t*(np.sin(th_t)-np.sin(th_t_1))
        y_t = y_t_1 - v_t/om_t*(np.cos(th_t)-np.cos(th_t_1))

        g = np.array([x_t,y_t,th_t])

        Gx = np.array([[1.,0.,v_t/om_t*(np.cos(th_t)-np.cos(th_t_1))],[0.,1.,v_t/om_t*(np.sin(th_t)-np.sin(th_t_1))],[0.,0.,1.]])

        Gu_om1 = v_t / (np.square(om_t)) * (np.sin(th_t_1) - np.sin(th_t)) + v_t * dt / om_t * np.cos(th_t)
        Gu_om2 = v_t / (np.square(om_t)) * (np.cos(th_t) - np.cos(th_t_1)) + v_t * dt / om_t * np.sin(th_t)
        Gu = np.array([[1/om_t*(np.sin(th_t)-np.sin(th_t_1)),Gu_om1],[- 1/om_t*(np.cos(th_t)-np.cos(th_t_1)),Gu_om2],[0.,dt]])

#    print("x_t,y_t,th_t",x_t,y_t,th_t)
#    print("g,Gx,Gu",g,Gx,Gu)


    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

#    print("alpha, r",alpha, r)

    x, y, th = x
    x_base, y_base, th_base = tf_base_to_camera

    x_cam = x + x_base*np.cos(th) - y_base*np.sin(th)
    y_cam = y + x_base*np.sin(th) + y_base*np.cos(th)
    th_cam = th + th_base

#    print("x_cam,y_cam,th_cam",x_cam,y_cam,th_cam)

#    r_in_cam = r - (x_cam/np.cos(th_cam))*np.cos(th_cam-alpha)
    alpha_in_cam = alpha - th_cam
#    r_in_cam = r - (np.sqrt(np.square(x_cam)+np.square(y_cam)))*np.cos(alpha-np.arctan2(y_cam, x_cam))
    r_in_cam = r - x_cam*np.cos(alpha)-y_cam*np.sin(alpha)

#    print("debug", (r - (np.sqrt(np.square(x_cam)+np.square(y_cam)))*np.cos(alpha-np.arctan2(y_cam, x_cam))), (r - x_cam*np.cos(alpha)-y_cam*np.sin(alpha)))

#    print("alpha_in_cam,r_in_cam",alpha_in_cam,r_in_cam)

    h=np.array([alpha_in_cam,r_in_cam])

    dr_dx = -np.cos(alpha)
    dr_dy = -np.sin(alpha)
#    dr_dthcam = x_cam * (np.sin(alpha_in_cam)*np.cos(th_cam)+np.cos(alpha_in_cam)*np.sin(th_cam))-y_cam*(np.cos(alpha_in_cam)*np.cos(th_cam)-np.sin(alpha_in_cam)*np.sin(th_cam))
#    dr_dthcam = x_cam * (np.sin(th_cam)*np.cos(alpha_in_cam)+np.cos(th_cam)*np.sin(alpha_in_cam))-y_cam*(np.cos(th_cam)*np.cos(alpha_in_cam)-np.sin(th_cam)*np.sin(alpha_in_cam))
    dr_dth = (x_base*np.cos(alpha)+y_base*np.sin(alpha))*np.sin(th) + (y_base*np.cos(alpha)-x_base*np.sin(alpha))*np.cos(th)

    Hx = np.array([[0.,0.,-1.],[dr_dx,dr_dy,dr_dth]])

#    print("h, Hx",h, Hx)

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
