import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
import plotly.io as pio
pio.renderers.default = 'browser'

def tbp(t, y, mu, case):
    x = y[0:3]
    v = y[3:6]

    if case == 'no':
        drdt = v
        dvdt = - mu * x / np.linalg.norm(x)**3

    elif case == 'j2':
        J2 = 0.1082626925638815e-2
        we = np.deg2rad(15.04/3600)
        w_e = np.array([0, 0, we])
        R_e = 6378.137  # km

        xx = np.linalg.norm(x)
        c = 1.5 * J2 * mu * R_e**2 / xx**4

        a_j2_x = c * x[0] / xx * (5*x[2]**2 / xx**2 - 1)
        a_j2_y = c * x[1] / xx * (5*x[2]**2 / xx**2 - 1)
        a_j2_z = c * x[2] / xx * (5*x[2]**2 / xx**2 - 3)

        drdt = v

        dvdt_x = - mu * x[0] / np.linalg.norm(x)**3 + a_j2_x
        dvdt_y = - mu * x[1] / np.linalg.norm(x)**3 + a_j2_y
        dvdt_z = - mu * x[2] / np.linalg.norm(x)**3 + a_j2_z

        dvdt = np.array([dvdt_x, dvdt_y, dvdt_z])

    return np.concatenate([drdt, dvdt])

def car2kep(R, V, mu):
    """
    Computes the Keplerian orbital elements from the state vector (R, V).
    
    Inputs:
    R  : Position vector [km]
    V  : Velocity vector [km/s]
    mu : Gravitational parameter [km^3/s^2]
    
    Outputs:
    a, e, i, OM, om, theta
    """
    eps = 1e-10
    
    # 1. Magnitudes and Radial Velocity
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R, V) / r
    
    # 2. Angular Momentum
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    
    # 3. Inclination (rad)
    # H[2] is the Z-component (MATLAB H(3))
    i = np.acos(H[2] / h)
    
    # 4. Node Vector
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)
    
    # 5. Right Ascension of Ascending Node (OM)
    if n != 0:
        OM = np.acos(N[0] / n) # N[0] is X-component
        if N[1] < 0:           # N[1] is Y-component
            OM = 2 * np.pi - OM
    else:
        OM = 0
        
    # 6. Eccentricity Vector and Magnitude
    E = 1 / mu * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)
    
    # 7. Argument of Perigee (om)
    if n != 0:
        if e > eps:
            om = np.acos(np.dot(N, E) / (n * e))
            if E[2] < 0: # E[2] is Z-component
                om = 2 * np.pi - om
        else:
            om = 0
    else:
        om = 0
        
    dum = np.dot(E, R) / (e * r)

    # Safety Check: ensure dum is within [-1, 1] to avoid math errors with acos
    dum = np.clip(dum, -1.0, 1.0)

    if e > eps:
        theta = np.acos(dum)
        
        # Quadrant check using radial velocity (vr)
        if vr < 0:
            theta = 2 * np.pi - theta
            
    else:
        # Circular orbit case: use the node vector N as a reference
        cp = np.cross(N, R)
        
        # (Note: dum is recalculated here in your MATLAB script, 
        # but it's the same math as above)
        if cp[2] >= 0: # cp[2] is the Z-component (MATLAB's cp(3))
            theta = np.acos(dum)
        else:
            theta = 2 * np.pi - np.acos(dum)

    # 9. Semi-major Axis
    a = h**2 / mu / (1 - e**2)

    kep_state = np.array([a, e, i, OM, om, theta])
    
    return kep_state

def kep2car(kep_state, mu):
    """
    Converts Keplerian orbital elements to Cartesian state vectors.
    
    Inputs:
    kep_state : Array of Keplerian elements [a, e, i, OM, om, theta]
    mu        : Gravitational parameter [km^3/s^2]
    
    Outputs:
    R : Position vector [km]
    V : Velocity vector [km/s]
    """
    a, e, i, OM, om, theta = kep_state

    # 1. Semi-latus rectum
    p = a * (1 - e**2)

    # 2. Position in perifocal coordinates
    r_perifocal = (p / (1 + e * np.cos(theta))) * np.array([np.cos(theta), np.sin(theta), 0])

    # 3. Velocity in perifocal coordinates
    v_perifocal = np.sqrt(mu / p) * np.array([-np.sin(theta), e + np.cos(theta), 0])

    # 4. Rotation matrices
    R3_OM = np.array([[np.cos(OM), -np.sin(OM), 0],
                      [np.sin(OM),  np.cos(OM), 0],
                      [0,           0,          1]])

    R1_i = np.array([[1, 0,           0],
                     [0, np.cos(i), -np.sin(i)],
                     [0, np.sin(i),  np.cos(i)]])

    R3_om = np.array([[np.cos(om), -np.sin(om), 0],
                      [np.sin(om),  np.cos(om), 0],
                      [0,           0,          1]])

    # Combined rotation matrix
    Q_pX = R3_OM @ R1_i @ R3_om

    # 5. Position and velocity in inertial frame
    R = Q_pX @ r_perifocal
    V = Q_pX @ v_perifocal

    return R, V

def add_earth_surface(fig, Re, opacity=0.18, n=50):
    phi, theta = np.mgrid[0:2*np.pi:complex(n), 0:np.pi:complex(n)]
    x = Re * np.cos(phi) * np.sin(theta)
    y = Re * np.sin(phi) * np.sin(theta)
    z = Re * np.cos(theta)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Greens',
        opacity=opacity,
        showscale=False,
        hoverinfo='skip',
        name='Earth'
    ))

R_M = 1_737_400

