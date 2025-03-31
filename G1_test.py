import numpy as np

import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
# franka = scene.add_entity(
#     gs.morphs.MJCF(
#         file="xml/franka_emika_panda/panda.xml",
#     ),
# )
g1 = scene.add_entity(
    gs.morphs.MJCF(
        file="g1_description/g1_29dof_rev_1_0.xml",
    ),
)
########################## build ##########################
scene.build()

jnt_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]
dofs_idx = [g1.get_joint(name).dof_idx_local for name in jnt_names]
############ Optional: set control gains ############
# set positional gains
g1.set_dofs_kp(
    kp=np.array([3000, 3000, 3000, 3000, 500, 500, 3000, 3000, 3000, 3000, 500, 500, 300, 300, 300, 1000, 1000, 1000, 500, 100, 100, 100, 1000, 1000, 1000, 500, 100, 100, 100]),
    dofs_idx_local=dofs_idx,
)
# set velocity gains
g1.set_dofs_kv(
    kv=np.array([300, 300, 300, 300, 50, 50, 300, 300, 300, 300, 50, 50, 30, 30, 30, 100, 100, 100, 50, 10, 10, 10, 100, 100, 100, 50, 10, 10, 10 ]),
    dofs_idx_local=dofs_idx,
)
# set force range for safety
g1.set_dofs_force_range(
    lower=np.array([-88, -139, -88, -139, -50, -50, -88, -139, -88, -139, -50, -50, -88, -50, -50, -25, -25, -25, -25, -25, -5, -5, -25, -25, -25, -25, -25, -5, -5 ]),
    upper=np.array([88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5 ]),
    dofs_idx_local=dofs_idx,
)
# # Hard reset
# for i in range(150):
#     if i < 50:
#         franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
#     elif i < 100:
#         franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
#     else:
#         franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)
#
#     scene.step()

# PD control
for i in range(5000):
    # if i == 0:
    #     g1.control_dofs_position(
    #         np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
    #         dofs_idx,
    #     )
    # elif i == 250:
    #     g1.control_dofs_position(
    #         np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
    #         dofs_idx,
    #     )
    # elif i == 500:
    #     g1.control_dofs_position(
    #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #         dofs_idx,
    #     )
    # elif i == 750:
    #     # control first dof with velocity, and the rest with position
    #     g1.control_dofs_position(
    #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
    #         dofs_idx[1:],
    #     )
    #     g1.control_dofs_velocity(
    #         np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
    #         dofs_idx[:1],
    #     )
    # elif i == 1000:
    #     g1.control_dofs_force(
    #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #         dofs_idx,
    #     )
    if i == 0:
        g1.control_dofs_position(
            np.zeros(29),
            dofs_idx,
        )
    elif i == 250:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 500:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 750:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 1000:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 1250:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 1500:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 1750:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 2000:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )
    elif i == 2250:
        g1.control_dofs_position(
            np.random.rand(29),
            dofs_idx,
        )

    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    print("control force:", g1.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    print("internal force:", g1.get_dofs_force(dofs_idx))

    scene.step()
