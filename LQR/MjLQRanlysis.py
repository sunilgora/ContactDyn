import mujoco as mj
import mujoco.viewer
import numpy as np
import os,sys,time
import matplotlib.pyplot as plt
from mujoco.glfw import glfw
from myContact import trnparam, DepthvsForce, mjforce, sysident, runFwdDyn

import pickle
import scipy
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk
import threading
import queue

xml_path= 'humanoid.xml'

# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

mujoco.mj_forward(model, data)

for key in range(model.nkey):
  mujoco.mj_resetDataKeyframe(model, data, key)
  mujoco.mj_forward(model, data)

DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Initialize to the standing-on-one-leg pose.
mujoco.mj_resetDataKeyframe(model, data, 1)

# Skip the initial viewer loop - we'll start with the GUI directly


#Initial contact force analysis

mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0  # Assert that there is no the acceleration.
mujoco.mj_inverse(model, data)
print(data.qfrc_inverse)

height_offsets = np.linspace(-0.001, 0.001, 2001)
vertical_forces = []
for offset in height_offsets:
  mujoco.mj_resetDataKeyframe(model, data, 1)
  mujoco.mj_forward(model, data)
  data.qacc = 0
  # Offset the height by `offset`.
  data.qpos[2] += offset
  mujoco.mj_inverse(model, data)
  vertical_forces.append(data.qfrc_inverse[2])

# Find the height-offset at which the vertical force is smallest.
idx = np.argmin(np.abs(vertical_forces))
best_offset = height_offsets[idx]

# Plot the relationship.
plt.figure(figsize=(10, 6))
plt.plot(height_offsets * 1000, vertical_forces, linewidth=3)
# Red vertical line at offset corresponding to smallest vertical force.
plt.axvline(x=best_offset*1000, color='red', linestyle='--')
# Green horizontal line at the humanoid's weight.
weight = model.body_subtreemass[1]*np.linalg.norm(model.opt.gravity)
plt.axhline(y=weight, color='green', linestyle='--')
plt.xlabel('Height offset (mm)')
plt.ylabel('Vertical force (N)')
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.title(f'Smallest vertical force '
          f'found at offset {best_offset*1000:.4f}mm.')
plt.show()

mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()  # Save the position setpoint.
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()
print('desired forces:', qfrc0)

actuator_moment = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(
    actuator_moment,
    data.actuator_moment.reshape(-1),
    data.moment_rownnz,
    data.moment_rowadr,
    data.moment_colind.reshape(-1),
)
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
print('control setpoint:', ctrl0)

data.ctrl = ctrl0
mujoco.mj_forward(model, data)
print('actuator forces:', data.qfrc_actuator)

# Set the state and controls to their setpoints.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
data.ctrl = ctrl0

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(1/FRAMERATE)

# GUI for updating geom_solref parameter
class SolrefGUI:
    def __init__(self, model, data, qpos0, ctrl0, K):
        self.model = model
        self.data = data
        self.qpos0 = qpos0
        self.ctrl0 = ctrl0
        self.K = K
        self.running = False
        self.viewer = None
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Create main window
        self.root = tk.Tk()
        self.root.title("MuJoCo LQR Controller")
        self.root.geometry("600x200+50+50")  # Position at (50,50) with size 600x500
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create matplotlib figure for controls
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.subplots_adjust(bottom=0.4)

        # Current value display
        self.value_text = self.ax.text(0.5, 0.7, f'geom_solimp[0]: {model.geom_solimp[0]}',
                                     transform=self.ax.transAxes, ha='center', fontsize=12)
        self.value_text = self.ax.text(0.5, 0.5, f'geom_solref[0]: {model.geom_solref[0]}',
                                     transform=self.ax.transAxes, ha='center', fontsize=12)
        
        # Create slider
        ax_slider = self.fig.add_axes([0.3, 0.2, 0.55, 0.03])
        self.slider = Slider(ax_slider, 'Time constant', 0.0001, 1.0, valinit=model.geom_solref[0][0])
        self.slider.on_changed(self.update_solref)

        # Status text
        # self.status_text = self.ax.text(0.5, 0.95, 'Simulation running...',
        #                               transform=self.ax.transAxes, ha='center', fontsize=10)

        # # Instructions
        # instructions = """
        # Real-time Contact Stiffness Control
        # ===================================
        
        # • The slider below controls geom_solref[0][0] in real-time
        # • Move the slider to adjust contact stiffness instantly
        # • Watch the humanoid's balance change immediately
        # • Simulation starts automatically - MuJoCo viewer opens separately
        # • Use the Stop button or close windows to end simulation
        # """
        # self.ax.text(0.5, 0.35, instructions, transform=self.ax.transAxes,
        #             ha='center', fontsize=8, wrap=True, linespacing=1.5)

        # Hide axes
        self.ax.set_axis_off()

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control buttons
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.stop_button = ttk.Button(self.button_frame, text="Stop Simulation",
                                    command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # Start simulation automatically
        self.start_simulation()

    def update_solref(self, value):
        # Update the geom_solref parameter
        self.model.geom_solref[0][0] = float(value)
        self.value_text.set_text(f'Current geom_solref[0]: {self.model.geom_solref[0]}')
        self.canvas.draw()

    def start_simulation(self):
        # self.status_text.set_text('Simulation running...')
        self.canvas.draw()
        self.running = True

        # Start simulation in a separate thread
        sim_thread = threading.Thread(target=self.run_simulation)
        sim_thread.daemon = True
        sim_thread.start()

    def stop_simulation(self):
        self.running = False
        self.command_queue.put("stop")
        # self.status_text.set_text('Simulation stopped')
        self.canvas.draw()

    def run_simulation(self):
        try:
            # Reset data, set initial pose
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos = self.qpos0
            self.data.ctrl = self.ctrl0

            # Allocate position difference dq
            dq = np.zeros(self.model.nv)

            # Signal that we're ready
            self.response_queue.put("started")

            with mujoco.viewer.launch_passive(self.model, self.data,show_left_ui=False,show_right_ui=False) as viewer:
                self.viewer = viewer

                while viewer.is_running() and self.running:
                    # Check for commands
                    try:
                        cmd = self.command_queue.get_nowait()
                        if cmd == "stop":
                            break
                    except queue.Empty:
                        pass

                    # Get state difference dx
                    mujoco.mj_differentiatePos(self.model, dq, 1, self.qpos0, self.data.qpos)
                    dx = np.hstack((dq, self.data.qvel)).T

                    # LQR control law
                    self.data.ctrl = self.ctrl0 - self.K @ dx
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(1/FRAMERATE)
        except Exception as e:
            print(f"Simulation error: {e}")
            self.response_queue.put(f"error: {e}")
        finally:
            self.running = False
            self.viewer = None
            self.response_queue.put("stopped")

    def check_queues(self):
        # Check for responses from simulation thread
        try:
            while True:
                response = self.response_queue.get_nowait()
                if response == "started":
                    self.status_text.set_text('Simulation running - adjust slider to change stiffness in real-time')
                    self.canvas.draw()
                elif response == "stopped":
                    self.status_text.set_text('Simulation stopped')
                    self.canvas.draw()
                elif response.startswith("error:"):
                    self.status_text.set_text(f"Error: {response[6:]}")
                    self.canvas.draw()
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_queues)

    def on_closing(self):
        self.stop_simulation()
        plt.close(self.fig)
        self.root.destroy()

#LQR 

nu = model.nu  # Alias for the number of actuators.
R = np.eye(nu)

nv = model.nv  # Shortcut for the number of DoFs.

# Get the Jacobian for the root body (torso) CoM.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
mujoco.mj_forward(model, data)
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)

# Get the Jacobian for the left foot.
jac_foot = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('foot_left').id)

jac_diff = jac_com - jac_foot
Qbalance = jac_diff.T @ jac_diff

# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'abdomen' in name
    and not 'z' in name
]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'left' in name
    and ('hip' in name or 'knee' in name or 'ankle' in name)
    and not 'z' in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Cost coefficients.
BALANCE_COST        = 1000  # Balancing.
BALANCE_JOINT_COST  = 3     # Joints required for balancing.
OTHER_JOINT_COST    = .3    # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

# Create and run the GUI for controlling geom_solref
gui = SolrefGUI(model, data, qpos0, ctrl0, K)
gui.root.mainloop()

