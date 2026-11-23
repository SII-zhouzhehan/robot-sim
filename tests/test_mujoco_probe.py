"""Physical connectivity must survive rigid frames introduced by CAD export."""
import mujoco

from exploration.mujoco_probe import physical_joint_edges


def test_joint_graph_collapses_fixed_frames_and_keeps_branches():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="base">
          <body name="fixed_mount">
            <body name="arm_a">
              <joint name="a" axis="0 0 1"/>
              <geom type="sphere" size=".1"/>
              <body name="fixed_spacer" pos=".2 0 0">
                <body name="arm_b">
                  <joint name="b" axis="0 0 1"/>
                  <geom type="sphere" size=".1"/>
                </body>
              </body>
            </body>
          </body>
          <body name="arm_c" pos="0 .3 0">
            <joint name="c" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in ("a", "b", "c")]
    assert physical_joint_edges(model, ids) == [(0, 1), (0, 2)]
