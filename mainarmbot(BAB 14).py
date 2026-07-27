import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import math

def setup_pybullet():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    table_z = 0.2
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.2])
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.2], rgbaColor=[0.2, 0.2, 0.2, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=[0.5, 0, table_z])

def load_arm_robot():
    robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.4], useFixedBase=True)
    home_joints = [0, 0, 0, 1.57, 0, -1.57, 0]
    for i in range(7):
        p.resetJointState(robot, i, home_joints[i])
    return robot, 6

def load_objects():
    start_pos = [0.5, 0.3, 0.425]
    target_pos = [0.5, -0.3, 0.425]
    cube = p.loadURDF("cube_small.urdf", start_pos)
    p.addUserDebugLine([0.5, -0.3, 0.4], [0.5, -0.3, 0.9], [1, 0, 0], 3, 2)
    return cube, start_pos, target_pos

def setup_ceiling_camera():
    width, height = 320, 240
    fov, aspect, near, far = 57, width/height, 0.1, 5.0
    cam_pos = [0.5, 0.0, 1.5]
    target_pos = [0.5, 0.0, 0.4]
    up_vector = [0, 1, 0]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    return width, height, near, far, view_matrix, proj_matrix

def capture_camera_frames(cam_params):
    w, h, near, far, view_mat, proj_mat = cam_params
    _, _, px, depth_buf, _ = p.getCameraImage(w, h, view_mat, proj_mat)
    
    rgb_array = np.array(px, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.putText(rgb_bgr, "Ceiling RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    depth_np = np.array(depth_buf).reshape(h, w)
    depth_m = far * near / (far - (far - near) * depth_np)
    depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.putText(depth_colored, "Ceiling Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    ir_gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    ir_colored = cv2.applyColorMap(ir_gray, cv2.COLORMAP_TWILIGHT)
    cv2.putText(ir_colored, "Ceiling IR (Sim)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return rgb_bgr, depth_colored, ir_colored

def get_kinematics_info(robot_id, ee_link, target_pos):
    state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    fk_pos = state[4]
    fk_orn = p.getEulerFromQuaternion(state[5])
    joint_angles = [round(p.getJointState(robot_id, j)[0], 2) for j in range(7)]
    
    info_text = (
        f"--- KINEMATICS INFO ---\n"
        f"DOF: 7 Axis\n"
        f"Target XYZ: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]\n"
        f"FK Pos XYZ: [{fk_pos[0]:.2f}, {fk_pos[1]:.2f}, {fk_pos[2]:.2f}]\n"
        f"FK Orn RPY: [{math.degrees(fk_orn[0]):.1f}, {math.degrees(fk_orn[1]):.1f}, {math.degrees(fk_orn[2]):.1f}]\n"
        f"Joints: {joint_angles}"
    )
    return info_text

def render_external_gui(rgb, depth, ir, kin_text):
    combined_cameras = np.hstack((rgb, depth, ir))
    h, w, _ = combined_cameras.shape
    text_h = 150
    text_canvas = np.zeros((text_h, w, 3), np.uint8)
    
    y0, dy = 20, 20
    for i, line in enumerate(kin_text.split('\n')):
        y = y0 + i * dy
        cv2.putText(text_canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    final_gui = np.vstack((combined_cameras, text_canvas))
    cv2.imshow('External Dashboard (Cameras & Kinematics)', final_gui)

def move_arm(robot_id, ee_link, target_pos, target_orn, cam_params, steps=100):
    joint_poses = p.calculateInverseKinematics(robot_id, ee_link, target_pos, target_orn)
    
    for i in range(steps):
        for j in range(7):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, joint_poses[j], force=50)
        p.stepSimulation()
        
        rgb, depth, ir = capture_camera_frames(cam_params)
        kin_text = get_kinematics_info(robot_id, ee_link, target_pos)
        render_external_gui(rgb, depth, ir, kin_text)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        time.sleep(1./240.)
    return False

def execute_pick_and_place():
    setup_pybullet()
    robot, ee_link = load_arm_robot()
    cube, cube_start, cube_target = load_objects()
    cam_params = setup_ceiling_camera()
    target_orn = p.getQuaternionFromEuler([0, math.pi, 0])
    constraint_id = -1
    step = 0
    
    while True:
        if step == 0:
            if move_arm(robot, ee_link, [cube_start[0], cube_start[1], 0.6], target_orn, cam_params): break
            step = 1
        elif step == 1:
            if move_arm(robot, ee_link, [cube_start[0], cube_start[1], 0.43], target_orn, cam_params): break
            step = 2
        elif step == 2:
            constraint_id = p.createConstraint(robot, ee_link, cube, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,-0.025])
            step = 3
        elif step == 3:
            if move_arm(robot, ee_link, [cube_start[0], cube_start[1], 0.9], target_orn, cam_params): break
            step = 4
        elif step == 4:
            if move_arm(robot, ee_link, [cube_target[0], cube_target[1], 0.9], target_orn, cam_params): break
            step = 5
        elif step == 5:
            if move_arm(robot, ee_link, [cube_target[0], cube_target[1], 0.43], target_orn, cam_params): break
            step = 6
        elif step == 6:
            if constraint_id != -1:
                p.removeConstraint(constraint_id)
            step = 7
        elif step == 7:
            if move_arm(robot, ee_link, [0.0, 0.0, 1.1], target_orn, cam_params): break
            p.resetBasePositionAndOrientation(cube, cube_start, [0,0,0,1])
            step = 0
            time.sleep(2)

        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    execute_pick_and_place()