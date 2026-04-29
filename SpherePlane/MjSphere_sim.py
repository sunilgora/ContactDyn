#======== Sphere-Plane Contact with Default MuJoCo Contact Model ========#
import mujoco
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

#Plot set
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
# To ensure correct rendering of minus signs in mathematical expressions
plt.rcParams["axes.formatter.use_mathtext"] = True        
plt.rcParams['pdf.fonttype'] = 42  # avoid type3 font
# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# Set the default linewidth to 2
plt.rcParams['lines.linewidth'] = 3
fsize=16
plt.rc('font', size=fsize)  # controls default text sizes
plt.rc('axes', titlesize=fsize)  # fontsize of the axes title
plt.rc('axes', labelsize=fsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('legend', fontsize=fsize)  # legend fontsize
plt.rc('figure', titlesize=fsize+2)  # fontsize of the figure title
plt.rcParams["axes.spines.right"] = "False"
plt.rcParams["axes.spines.top"] = "False"
plt.rcParams['axes.autolimit_mode'] = 'round_numbers' # to avoid offset in axis
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

# Path to the XML file
xml_path = '/home/sg/Documents/Mdrive/Python/MuJoCo/Contact/SpherePlane/sphere.xml'

# Load model once
model = mujoco.MjModel.from_xml_path(xml_path)
# print('Modified stiffness for r/2 deformation:',1/4*(1-0.475)*9.81*(0.95)**2/(0.006*0.475**2))

def simulate(dampratio, static_def, midpt, power):
    #Modify solref and solimp
    # Change solref for the plane geom (assuming geom 1 is the plane)
    # static_def=0.006
    d0=0
    dwidth=0.95
    # midpt=0.5
    width=static_def/midpt
    # power=5
    model.geom_solimp[0] = np.array([d0, dwidth, width, midpt, power])
    ymidpt=midpt #(1/midpt)**(power-1)*(midpt**power)
    dmidpt=d0+ymidpt*(dwidth-d0)
    stiffness = (1-dmidpt)*9.81*(dwidth**2)/(static_def*dmidpt**2)
    damping = 2 * dampratio * np.sqrt(stiffness)
    print('siffness:', stiffness, 'damping:', damping)
    model.geom_solref[0] = np.array([-stiffness, -damping])
    
    data = mujoco.MjData(model)
    
    # Set initial velocity downward for faster drop
    # data.qvel[2] = -0.1  # downward velocity
    
    deformations = []
    forces = []
    
    for _ in range(50000):  # Simulate for 50000 steps
        mujoco.mj_step(model, data)
        z = data.qpos[2]
        deformation = max(0, 0.01 - z)  # radius is 0.01
        if len(data.contact) > 0:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, 0, contact_force)
            force_val = contact_force[0]  # z-component of force
            deformations.append(deformation)
            forces.append(force_val)
    
    return deformations, forces

def InvDyn(dampratio, timeconst):
    # Change solref for the plane geom (assuming geom 1 is the plane)
    model.geom_solref[0] = np.array([timeconst, dampratio])
    
    data = mujoco.MjData(model)
    
    # Set initial velocity downward for faster drop
    # data.qvel[2] = -0.1  # downward velocity
    
    deformations = []
    forces = []
    
    for _ in range(50000):  # Simulate for 50000 steps
        mujoco.mj_inverse(model, data)
        z = data.qpos[2]
        deformation = max(0, 0.01 - z)  # radius is 0.01
        if len(data.contact) > 0:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, 0, contact_force)
            force_val = contact_force[0]  # z-component of force
            deformations.append(deformation)
            forces.append(force_val)
    
    return deformations, forces

# GUI setup
# plt.style.use('seaborn-v0_8')  # Use a nice style
fig, ax = plt.subplots(figsize=(5, 8))
plt.subplots_adjust(bottom=0.35, top=0.95, left=0.15, right=0.9)

# ax.set_title('Force vs Deformation in MuJoCo sphere Simulation')
weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
ax.axhline(y=weight, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Deformation (m)')
ax.set_ylabel('Contact Force (N)')
ax.grid(True, alpha=0.3)

# Slider for deformation 
ax_slider_time = plt.axes([0.35, 0.15, 0.5, 0.03])
slider_def = Slider(ax_slider_time, 'Deformation', 0.001, 0.01, valinit=0.01, valstep=0.001)

# Slider for damping ratio
ax_slider_damp = plt.axes([0.35, 0.1, 0.5, 0.03])
slider_damp = Slider(ax_slider_damp, 'Damping Ratio', 0.01, 2.0, valinit=1.0, valstep=0.01)

# Slider for midpoint
ax_slider_midpt = plt.axes([0.35, 0.05, 0.5, 0.03])
slider_midpt = Slider(ax_slider_midpt, 'Midpoint', 0.01, 0.99, valinit=0.5, valstep=0.01)

# Slider for power
ax_slider_power = plt.axes([0.35, 0.0, 0.5, 0.03])
slider_power = Slider(ax_slider_power, 'Power', 0.5, 10, valinit=2, valstep=0.1)

# Button to plot
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Plot')

def plot_func(event):
    static_def = slider_def.val
    dampratio = slider_damp.val
    midpt = slider_midpt.val
    power = slider_power.val
    deps, fors = simulate(dampratio, static_def, midpt, power)
    ax.plot(deps, fors, label=f'solimp: [0, 0.95, {static_def/midpt:.3f}, {midpt:.2f}, {power:.2f}]', linewidth=2)
    ax.legend(loc='upper right')
    plt.draw()
    #save figure
    #Adjust margins and save
    fig.savefig(f'MjSphere.pdf', dpi=300)

button.on_clicked(plot_func)
plt.show()