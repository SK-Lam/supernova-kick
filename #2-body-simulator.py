#2-body-simulator

import numpy as np
import scipy as sci
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


#Universal Gravitation Constant
G=6.67408e-11 #N-m2/kg2

m_nd=1.989e+30 #kg #solar mass
# r_nd=1.500e+12 #m #10 AU
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #speed of the earth orbiting around the sun
t_nd=79.91*365.25*24*3600 #s #100 years
# t_nd=100*365.25*24*3600 #s #100 years

#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd


def com_pos(M1,R1,M2,R2):
    return (M1*R1 + M2*R2)/(M1+M2)

def com_vel(M1,V1,M2,V2):
    return (M1*V1 + M2*V2)/(M1+M2)


#Defining masses (solar mass)
# m1 = 3 # Massive star
# m2 = 0.2 # Companion

m1 = 1.44 # Massive star
m2 = 1 # Companion


number_of_run = 15 # number of run
for run in range(number_of_run):

    #Defining initial position and velocity vectors
    # r1 = [-0.05,0,0] #m
    # r2 = [0.05,0,0] #m
    r1 = [-0.5,0,0] #m
    r2 = [0.5,0,0] #m
    v1 = [0.01,0.01,0] #m/s
    v2 = [-0.05,0,-0.1] #m/s


    #Converting pos and vel vectors to arrays
    r1 = np.array(r1) 
    r2 = np.array(r2) 
    v1 = np.array(v1)
    v2 = np.array(v2)

    #Find pos and vel COM
    r_com = com_pos(m1,r1,m2,r2)
    v_com = com_vel(m1,r1,m2,r2)

    # Add a random vector

    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    # costheta = 0

    theta = round(np.arccos( costheta ), 4)
    u_rand = round(np.random.uniform(-0.05,0.2), 3)
    x = np.sin( theta ) * np.cos( phi ) * u_rand
    y = np.sin( theta ) * np.sin( phi ) * u_rand
    z = np.cos( theta ) * u_rand
    v_rand = (x,y,z)
    # v_rand = (0,0,0) 
    v_rand = np.array(v_rand)
    #print(v_rand)

    v2 = v2 + v_rand # new initial velocity of m2

    #Defining func for the equations of motion
    def Two_Body_eqn(w,t,G,m1,m2):
        r1 = w[:3]
        r2 = w[3:6]
        v1 = w[6:9]
        v2 = w[9:12]
        
        r=sci.linalg.norm(r2-r1) #Calculating normal of vector
        
        #eqn
        a_1 = K1*m2*(r2-r1) / r**3 # a_1 is d(v1)/dt
        a_2 = K1*m1*(r1-r2) / r**3
        #eqn 
        v_1 = K2*v1 # v_1 is d(r1)/dt
        v_2 = K2*v2 # + v_rand
        
        r_derivs = np.concatenate((v_1, v_2))
        derivs = np.concatenate((r_derivs, a_1 ,a_2))
        return derivs

    time_step = 300

    #initial parameters
    init_params = np.array([r1,r2,v1,v2]).flatten()
    time_span = np.linspace(0,2,time_step) # no. of frame

    # ODE solver
    import scipy.integrate as si

    Two_Body_sol = si.odeint(Two_Body_eqn,init_params,time_span,args=(G,m1,m2))

    r1_sol = Two_Body_sol[:,:3]
    r2_sol = Two_Body_sol[:,3:6]

    # Make the figure
    # plt.rcParams['animation.ffmpeg_path'] = '/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ffmpeg/'
    # plt.rcParams['animation.ffmpeg_path'] = '/home/opt/local/bin/ffmpeg'
    # plt.rcParams['animation.ffmpeg_path'] = '/home/codespace/.python/current/lib/python3.10/'

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d') 

    #Create new arrays for animation, th
    #Currently set to select every 4th point
    r1_anim = r1_sol[::1,:].copy()
    r2_anim = r2_sol[::1,:].copy()

    #Set initial marker for planets
    h1 = [ax.scatter(r1_anim[0,0],r1_anim[0,1],r1_anim[0,2],color="darkblue",marker="o",s=80,label="Massive Star")]
    h2 = [ax.scatter(r2_anim[0,0],r2_anim[0,1],r2_anim[0,2],color="tab:red",marker="o",s=80,label="Companion Star")]

    ax.set_xlabel("x",fontsize=14)
    ax.set_ylabel("y",fontsize=14)
    ax.set_zlabel("z",fontsize=14)
    ax.set_title("Visualization of orbits of stars in a 2-body system\n",fontsize=16)
    ax.legend(loc="upper left",fontsize=14)


    # Creating a function Animate that changes plots every frame('i' - frame no.)
    def Animate_2b(i,head1,head2):
        #Remove old markers
        h1[0].remove()
        h2[0].remove()
        
        # Plotting the orbits (for every i, plot from init pos to final pos)
        t1 = ax.plot(r1_anim[:i,0],r1_anim[:i,1],r1_anim[:i,2],color='darkblue')
        t2 = ax.plot(r2_anim[:i,0],r2_anim[:i,1],r2_anim[:i,2],color='r')
        
        # Plotting the current markers

        h1[0]=ax.scatter(r1_anim[i,0],r1_anim[i,1],r1_anim[i,2],color="darkblue",marker="o",s=80)
        h2[0]=ax.scatter(r2_anim[i,0],r2_anim[i,1],r2_anim[i,2],color="r",marker="o",s=80)
        
        ax.set_xbound(lower=-2.5, upper=0.0)
        ax.set_ybound(lower=0.0, upper=1.2)
        ax.set_zbound(lower=-3.0, upper=0.0)

        return t1,t2,h1,h2
        
    anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=time_step,interval=2,repeat=False,blit=False,fargs=(h1,h2))
    # Using the function module to make the animation

    ###############

    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=4000)
    # FFwriter = animation.FFMpegWriter(fps=30) ############

    #To save animation to disk, enable this
    
    #anim_2b.save("Your Directory" + "Twobodytest_u_rand="+str(u_rand)+"theta="+str(theta)+".gif") ##########B
    #print("Twobodytest_u_rand="+str(u_rand)+"theta="+str(theta)+".gif is saved")
    