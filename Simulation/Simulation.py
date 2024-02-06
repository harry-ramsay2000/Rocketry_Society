import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Thrust Curve
thrust = pd.read_csv('Simulation/Thrust_curve.csv', skiprows=3)
m_propellant = 52/1000  # kg
t_interp = interp1d(thrust['Time (s)'], thrust['Thrust (N)'], kind='linear', fill_value=(0,0), bounds_error=False)
fuel_mass_rate =  m_propellant/thrust['Time (s)'].max()

# plt.plot(thrust['Time (s)'], thrust['Thrust (N)'])
# plt.xlabel('Time (s)')
# plt.ylabel('Thrust (N)')
# plt.title('Thrust Curve')
# plt.show()

# Rocket Parameters
m_dry = 1  # kg
# print('Total mass: ',m_dry + m_propellant)  # kg
area = np.pi * (0.029/2)**2  # m^2
chute_area = np.pi * 0.3**2  # m^2
Cd = 0.75  # drag coefficient

# Simulation Parameters
dt = 0.01  # s
simulation_time = 100  # s
t = np.arange(0, simulation_time, dt)

# Initial Conditions
acceleration = [0]
velocity = [0]
position = [0]
mass = [m_dry + m_propellant]
drag = [0]
chute = [False]

# Simulation
for i in range(len(t)-1):
    F_thrust = t_interp(t[i])
    F_drag = 0.5 * 1.225 * Cd * area * velocity[i]**2
    F_chute = 0.5 * 1.225 * Cd * chute_area * velocity[i]**2
    F_gravity = 9.80665 * mass[i]
    if mass[i] > m_dry:
        F_net = F_thrust - (F_gravity + F_drag)
        chute.append(False)
    else:
        if np.gradient(position)[-1]<=0:
            F_net = F_chute + F_drag - F_gravity
            chute.append(True)
        else:
            F_net = F_drag - F_gravity
            chute.append(False)

    a = F_net / mass[i]
    v = velocity[i] + (a * dt)
    x = np.max([position[i] + (v * dt), 0])
    m = np.max([mass[i] - (fuel_mass_rate * dt), m_dry])
    
    drag.append(F_drag)
    acceleration.append(a)
    velocity.append(v)
    position.append(x)
    mass.append(m)

xlim = t[np.where(np.gradient(position)==0)[0][1]]
print('Time to Apogee:', t[np.where(np.array(position)==np.max(position))[0][0]], 's')
print('Time to ground: ', xlim, 's')
print('Descent Rate: ', np.min(velocity), 'm/s')

fig, ax = plt.subplots(5, 1, dpi=150, figsize=(10, 10), sharex=True)

ax[0].plot(t, acceleration)
ax[0].axhline(np.max(acceleration), xmax=(22/30), color='r', linestyle='--')
ax[0].text(xlim, np.max(acceleration), f'Max Acceleration: {np.max(acceleration):.2f} m/s$^2$', fontsize=10, color='r', verticalalignment='top', ha='right')
ax[0].set_ylabel('Acceleration [m/s$^2$]')

ax[1].plot(t, velocity)
ax[1].axhline(np.max(velocity), xmax=(23/30), color='r', linestyle='--')
ax[1].text(simulation_time, np.max(velocity), f'Max Velocity: {np.max(velocity):.2f} m/s', fontsize=10, color='r', verticalalignment='top', ha='right')
ax[1].set_ylabel('Velocity [m/s]')

alt_feet = 1
ftm = 3.28084
ax[2].plot(t, np.array(position)*(alt_feet*ftm))
ax[2].axhline(np.max(position)*(alt_feet*ftm), xmax=(23/30), color='r', linestyle='--')
ax[2].axhline((2500/ftm)*(alt_feet*ftm), xmax=23/30, color='g', linestyle='--')
ax[2].text(simulation_time, (2500/ftm)*(alt_feet*ftm), '2500ft Target Altitude', fontsize=10, color='g', verticalalignment='top', ha='right')
if alt_feet == 1:
    ax[2].set_ylabel('Position [ft]')
    ax[2].text(simulation_time, np.max(position)*(alt_feet*ftm), f'Max Position: {(np.max(position)*(alt_feet*ftm)):.2f} ft', fontsize=10, color='r', verticalalignment='top', ha='right')
else:
    ax[2].text(simulation_time, np.max(position)*(alt_feet*ftm), f'Max Position: {(np.max(position)*(alt_feet*ftm)):.2f} m', fontsize=10, color='r', verticalalignment='top', ha='right')
    ax[2].set_ylabel('Position [m]')

ax[3].plot(t, mass)
ax[3].set_ylabel('Mass [kg]')

ax[4].plot(t, drag)
ax[4].set_ylabel('Drag [N]')
ax[4].axvline(t[np.where(chute)[0][0]], c='purple', linestyle='--', label='Chute Deployed', alpha=0.5)
ax[4].legend(frameon=False)

ax[4].set_xlabel('Time [s]')

# plt.show()
fig.tight_layout()
fig.savefig('Simulation/Simulation.png')


