# Code to create generic URDF from DH parameters

import numpy as np
from numpy import sin, cos, pi

from scipy.spatial.transform import Rotation as R

def DH_trans(DH, joint_val):

     d, theta, a, alpha = (0,0,0,0)

     if (DH[0] == 'r'):

         d, theta, a, alpha = (DH[1], joint_val, DH[2], DH[3])

     elif (DH[0] == 'p'):

         d, theta, a, alpha = (joint_val, DH[1], DH[2], DH[3])

     elif (DH[0] == 'f'):

         d, theta, a, alpha = (DH[1], DH[2], DH[3], DH[4])

     trans_mat = np.array([[cos(theta), -1*sin(theta)*cos(alpha), sin(theta)*sin(alpha),    a*cos(theta)],
                           [sin(theta), cos(theta)*cos(alpha),    -1*cos(theta)*sin(alpha), a*sin(theta)],
                           [0,          sin(alpha),               cos(alpha),               d           ],
                           [0,          0,                        0,                        1           ]])

     return trans_mat


def joint_transforms(DH_Params):

     transforms = []

     current_DOF = 0

     transforms.append(np.eye(4))

     for DH in DH_Params:

         if (DH[0] == 'r' or DH[0] == 'p'):
             transforms.append(DH_trans(DH, 0.0))
             current_DOF = current_DOF + 1

         else:
             transforms.append(DH_trans(DH, 0.0))

     return transforms

def joint_frames(transforms):

         joint_frames = [transforms[0]]

         for trans in transforms[1:]:

             joint_frames.append(joint_frames[-1] @ trans)

         return joint_frames

# DH Parameter Layout:
# ['r', d, a, alpha] for revolute joints
# ['p', theta, a, alpha] for prismatic joints
# ['f', d, theta, a, alpha] for fixed joints

def xml_string(DH_Params, scale=1, alpha=1.0):

    outstring = ""

    transforms = joint_transforms(DH_Params)

    frames = joint_frames(transforms)

    outstring += "<robot name='robot'>\n"

    outstring += f"\t<material name='blue'>\n\t\t<color rgba='0 0 0.8 {alpha}'/>\n\t</material>\n"
    outstring += f"\t<material name='red'>\n\t\t<color rgba='0.8 0 0 {alpha}'/>\n\t</material>\n"
    outstring += f"\t<material name='gray'>\n\t\t<color rgba='0.2 0.2 0.2 {alpha}'/>\n\t</material>\n"
    outstring += f"\t<material name='gray2'>\n\t\t<color rgba='0.4 0.4 0.4 {alpha}'/>\n\t</material>\n"

    for i in range(len(transforms) - 1):

        el = transforms[i] # local
        fr = frames[i] # world

        # We need to create a cylinder to represent the joint
        # If the index is not zero, connect it to the previous link
        # And a joint to connect it to the link
        # And a box to connect the joints

        # rpy = R.from_matrix(fr[0:3,0:3]).as_euler('XYZ')
        rpy = R.from_matrix(el[0:3,0:3]).as_euler('XYZ')

        outstring = outstring + "\t<link name='a{}'>\n".format(i)
        outstring = outstring + "\t\t<visual>\n"
        outstring = outstring + "\t\t\t<origin rpy='0 0 0' xyz='0 0 0'/>\n"#.format(rpy[0], rpy[1], rpy[2], el[0,3], el[1,3], el[2,3])
        outstring = outstring + "\t\t\t<geometry>\n"
        outstring = outstring + "\t\t\t\t<cylinder length='0.28' radius='0.115'/>\n"
        outstring = outstring + "\t\t\t</geometry>\n"
        outstring = outstring + "\t\t\t<material name='blue'/>\n"
        outstring = outstring + "\t\t</visual>\n"
        outstring = outstring + "\t</link>\n"

        # outstring += "\t<joint name='world_weld' type='fixed'>\n\t\t<parent link='world'/>\n\t\t<child link='a0'/>\n\t\t<origin rpy='0 0 0' xyz='0 0 0'/>\n\t</joint>\n"

        origins_vector = transforms[i + 1][0:3,3]
        origins_orn = transforms[i+1][0:3,0:3]

        origins_vector_norm = np.linalg.norm(origins_vector)

        cylinder_origin = origins_vector/2.0
        link_rot = R.from_rotvec(np.array([0.0,1.0,0.0])*np.arctan2(DH_Params[i][2],DH_Params[i][1]))
        a_origin = el[0:3,3]

        # rpy = R.from_matrix(link_rot.as_matrix()@origins_orn.T).as_euler('XYZ')
        # If not on the first transformation, fix the cylinder to the previous link
        if(i != 0):
            rpy = R.from_matrix(el[0:3,0:3]).as_euler('XYZ')
            outstring = outstring + "\t<joint name='fix_a{}_to_l{}' type='fixed'>\n".format(i, i-1)
            outstring = outstring + "\t\t<parent link='l{}'/>\n".format(i-1)
            outstring = outstring + "\t\t<child link='a{}'/>\n".format(i)
            outstring = outstring + "\t\t<origin rpy='{} {} {}' xyz='{} {} {}'/>\n".format(rpy[0], rpy[1], rpy[2], a_origin[0], a_origin[1], a_origin[2])
            outstring = outstring + "\t</joint>\n"

        # Add a cylinder that goes from the current origin to the next one
        rpy = link_rot.as_euler('XYZ')

        outstring = outstring + "\t<link name='l{}'>\n".format(i)
        outstring = outstring + "\t\t<visual>\n"
        outstring = outstring + "\t\t\t<origin rpy='{} {} {}' xyz='{} {} {}'/>\n".format(rpy[0], rpy[1], rpy[2], cylinder_origin[0], cylinder_origin[1], cylinder_origin[2])
        outstring = outstring + "\t\t\t<geometry>\n"
        outstring = outstring + "\t\t\t\t<cylinder length='{}' radius='0.08'/>\n".format(origins_vector_norm)
        outstring = outstring + "\t\t\t</geometry>\n"
        outstring = outstring + "\t\t\t<material name='gray2'/>\n"
        outstring = outstring + "\t\t</visual>\n"
        outstring = outstring + "\t\t<collision>\n"
        outstring = outstring + "\t\t\t<origin rpy='{} {} {}' xyz='{} {} {}'/>\n".format(rpy[0], rpy[1], rpy[2], cylinder_origin[0], cylinder_origin[1], cylinder_origin[2])
        outstring = outstring + "\t\t\t<geometry>\n"
        outstring = outstring + "\t\t\t\t<cylinder length='{}' radius='0.08'/>\n".format(origins_vector_norm)
        outstring = outstring + "\t\t\t</geometry>\n"
        outstring = outstring + "\t\t</collision>\n"
        outstring = outstring + "\t</link>\n"

        # Add the actual joint between the cylinder and link

        jointType = ""

        if (DH_Params[i][0] == 'r'):
            jointType = "revolute"
        elif (DH_Params[i][0] == 'p'):
            jointType = "prismatic"
        else:
            jointType = "fixed"


        outstring = outstring + "\t<joint name='move_l{}_from_a{}' type='{}'>\n".format(i, i, jointType)
        outstring = outstring + "\t\t<parent link='a{}'/>\n".format(i)
        outstring = outstring + "\t\t<child link='l{}'/>\n".format(i)
        # outstring = outstring + "\t\t<axis xyz='{} {} {}'/>\n".format(fr[0,2], fr[1,2], fr[2,2])
        # outstring = outstring + "\t\t<axis xyz='{} {} {}'/>\n".format(el[0,2], el[1,2], el[2,2])
        outstring = outstring + "\t\t<axis xyz='0 0 1'/>\n".format(0,0,1)
        outstring = outstring + "\t\t<limit lower='-3.2' upper='3.2'/>\n"
        outstring = outstring + "\t\t<origin rpy='0 0 0' xyz='0 0 0'/>\n"#.format(rpy[0], rpy[1], rpy[2], el[0,3], el[1,3], el[2,3])
        outstring = outstring + "\t</joint>\n"

    print(i)
    outstring += "\t<link name='ee_base'>\n"
    outstring += "\t\t<visual>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.27 0.17 0.0725'/></geometry>\n"
    outstring += "\t\t\t<material name='gray'/>\n"
    outstring += "\t\t</visual>\n"
    outstring += "\t\t<collision>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.27 0.17 0.0725'/></geometry>\n"
    outstring += "\t\t</collision>\n"
    outstring += "\t</link>\n"
    outstring += "\t<link name='ee_finger_1'>\n"
    outstring += "\t\t<visual>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.05 0.17 0.2'/></geometry>\n"
    outstring += "\t\t\t<material name='gray'/>\n"
    outstring += "\t\t</visual>\n"
    outstring += "\t\t<collision>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.05 0.17 0.2'/></geometry>\n"
    outstring += "\t\t</collision>\n"
    outstring += "\t</link>\n"
    outstring += "\t<link name='ee_finger_2'>\n"
    outstring += "\t\t<visual>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.05 0.17 0.2'/></geometry>\n"
    outstring += "\t\t\t<material name='gray'/>\n"
    outstring += "\t\t</visual>\n"
    outstring += "\t\t<collision>\n"
    outstring += "\t\t\t<origin rpy='0.0 0.0 0.0' xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t\t<geometry><box size='0.05 0.17 0.2'/></geometry>\n"
    outstring += "\t\t</collision>\n"
    outstring += "\t</link>\n"
    outstring += "\t<joint name='l{}_ee_base' type='fixed'>\n".format(i)
    outstring += "\t\t<parent link='l{}'/>\n".format(i)
    outstring += "\t\t<child link='ee_base'/>\n"
    outstring += "\t\t<axis xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t<origin rpy='{} {} {}' xyz='{} {} {}'/>\n".format(rpy[0], rpy[1], rpy[2], cylinder_origin[0]*2, cylinder_origin[1]*2, cylinder_origin[2]*2)
    outstring += "\t</joint>\n"
    outstring += "\t\t<joint name='ee_base_ee_finger_1' type='fixed'>\n"
    outstring += "\t\t<parent link='ee_base'/>\n"
    outstring += "\t\t<child link='ee_finger_1'/>\n"
    outstring += "\t\t<axis xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t<origin rpy='0.0 0.0 0' xyz='-0.11 0.0 0.08'/>\n"
    outstring += "\t</joint>\n"
    outstring += "\t<joint name='ee_base_ee_finger_2' type='fixed'>\n"
    outstring += "\t\t<parent link='ee_base'/>\n"
    outstring += "\t\t<child link='ee_finger_2'/>\n"
    outstring += "\t\t<axis xyz='0.0 0.0 0.0'/>\n"
    outstring += "\t\t<origin rpy='0.0 0.0 0' xyz='0.11 0.0 0.08'/>\n"
    outstring += "\t</joint>\n"

    outstring = outstring + "</robot>\n"

    return outstring

arm_name = '3R'
# arm_name = '4R'
# arm_name = '7R'


filename = f'./arms/robot_{arm_name}_no_obs.urdf'

f = open(filename, "w")

# d, a, alpha

DH_Params = []

if arm_name == '3R':
     DH_Params.append(['r', 0.0,    1.0,  0.0])
     DH_Params.append(['r', 0.0,    1.0,  0.0])
     DH_Params.append(['r', 0.0,    1.0,  0.0])

elif arm_name == '4R':
     DH_Params.append(['r', 0,    1.4142,  np.pi/2.0])
     DH_Params.append(['r', 1.0,  1.4142, -np.pi/2.0])
     DH_Params.append(['r', -1.0, 1.4142,  np.pi/2.0])
     DH_Params.append(['r', 0.5,  1.2247,  0.0])

elif arm_name == '7R':
     DH_Params.append(['r', 0.0,   0.17, -1.7104])
     DH_Params.append(['r', 1.67,  1.42, -1.9897])
     DH_Params.append(['r', -0.69, 1.42, -1.1519])
     DH_Params.append(['r', -1.77, 0.56, 0.87266])
     DH_Params.append(['r', 2.42,  1.32, -1.6057])
     DH_Params.append(['r', -0.38, 1.27, -1.615])
     DH_Params.append(['r', 0.95,  0.0,  0.0])


alpha = 1.0
f.write(xml_string(DH_Params, alpha=alpha))


f.close()
quit()

import pybullet as p

p.connect(p.GUI)
robot_id = p.loadURDF(filename, useFixedBase=True)


def draw_frame(position, orientation, length=0.5, duration=0):
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    p.addUserDebugLine(position,
                       position + rot_matrix[:,0] * length,
                       [1, 0, 0],
                       lifeTime=duration)
    p.addUserDebugLine(position,
                       position + rot_matrix[:,1] * length,
                       [0, 1, 0],
                       lifeTime=duration)
    p.addUserDebugLine(position,
                       position + rot_matrix[:,2] * length,
                       [0, 0, 1],
                       lifeTime=duration)

# Get the number of joints
num_joints = p.getNumJoints(robot_id)

# Draw frames for each joint
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    link_state = p.getLinkState(robot_id, i)
    draw_frame(link_state[4], link_state[5])

p.setRealTimeSimulation(0)
p.setTimeStep(0.02)
p.setGravity(0,0,-9.8)
import time
while True:
     p.stepSimulation()
     time.sleep(0.02)
