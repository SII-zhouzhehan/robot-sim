"""Manual position targets for the current closed-chain robot, in radians."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/my_robot/mujoco/tars_mjcf_grounded/scene.xml"
sys.path.insert(0, str(ROOT / "src"))
from utils.mujoco_control_panel import ControlPanel, ControlState

GROUPS = {"形变": range(1, 9), "舵轮转向": (10, 13, 16, 19), "轮轴转角": (11, 14, 17, 20)}


class Targets(ControlState):
    def __init__(self, model):
        super().__init__()
        self.values = {"reset": False, "angles": {}}
        self.limits = {}
        for numbers in GROUPS.values():
            for number in numbers:
                name = f"dof_{number}"
                actuator = model.actuator(name + "_position")
                joint = model.joint(name)
                if actuator.trnid[0] != joint.id:
                    raise ValueError(f"Unexpected transmission for {name}")
                limit = joint.range if joint.limited[0] else (-math.pi, math.pi)
                self.limits[name] = tuple(map(float, limit))
                self.values[name] = 0.0

    def command(self, values):
        if values == {"reset": True}:
            self.update({name: 0.0 for name in self.limits}, reset=True)
            return
        if not isinstance(values, dict) or any(k not in self.limits for k in values):
            raise ValueError("Unknown actuator")
        for k, val in values.items():
            lo, hi = self.limits[k]
            if type(val) not in (float, int) or not lo <= val <= hi:
                raise ValueError(f"{k} outside [{lo}, {hi}] rad")
        self.update(values)


def page(state):
    rows = []
    for label, numbers in GROUPS.items():
        rows.append(f"<h2>{label}</h2>")
        for n in numbers:
            key = f"dof_{n}"
            lo, hi = state.limits[key]
            rows.append(f'<label>{key} <input type="range" min="{lo}" max="{hi}" step="0.001" value="0" data-key="{key}"> <output></output></label>')
    return """<!doctype html><meta charset="utf-8"><title>TARS 手动控制</title>
<style>body{font:16px system-ui;max-width:650px;margin:24px auto}label{display:grid;grid-template-columns:70px 1fr 195px;gap:10px;padding:4px}output{font-variant-numeric:tabular-nums}</style>
<h1>TARS 闭链机器人</h1><p>目标 / 实测角度（rad）。连续轴面板限幅 ±π。</p>""" + ''.join(rows) + """
<button id="reset">重置初始姿态</button><span id="status"></span>
<script>
const sliders=[...document.querySelectorAll('input')];
async function send(v){const r=await fetch('/command',{method:'POST',body:JSON.stringify(v)});if(!r.ok)throw Error(await r.text())}
for(const e of sliders)e.oninput=()=>send({[e.dataset.key]:Number(e.value)}).catch(err=>statusMsg(err));
function statusMsg(t){document.querySelector('#status').textContent=t}
document.querySelector('#reset').onclick=async()=>{await send({reset:true});sliders.forEach(e=>e.value=0)};
setInterval(async()=>{try{const s=await(await fetch('/state')).json();for(const e of sliders){const k=e.dataset.key;e.nextElementSibling.textContent=s[k].toFixed(3)+' / '+(s.angles[k]??0).toFixed(3)}statusMsg(' 仿真 '+s.time.toFixed(1)+' s')}catch{statusMsg(' 等待仿真连接')}},180);
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-panel", action="store_true")
    args = parser.parse_args()
    if args.seconds is not None and not (math.isfinite(args.seconds) and args.seconds > 0):
        parser.error("--seconds must be positive")
    if args.headless and args.seconds is None:
        parser.error("headless mode needs --seconds")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = Targets(model)
    names = list(state.limits)
    ids = [model.actuator(k + "_position").id for k in names]
    qadr = [model.joint(k).qposadr[0] for k in names]
    ticks = 0
    with ExitStack() as stack:
        viewer = None
        if not args.headless:
            import mujoco.viewer as mv
            viewer = stack.enter_context(mv.launch_passive(model, data, show_left_ui=False, show_right_ui=False))
            with viewer.lock():
                viewer.cam.lookat[:] = data.subtree_com[0]
                viewer.cam.distance = 1.35
                viewer.cam.azimuth = 130
                viewer.cam.elevation = -23
        if viewer and not args.no_panel:
            panel = ControlPanel(state, page=page(state))
            stack.callback(panel.close)
            print("Panel", panel.url, flush=True)
        while (viewer is None or viewer.is_running()) and (args.seconds is None or ticks * model.opt.timestep < args.seconds):
            start = time.monotonic()
            if state.consume_reset():
                mujoco.mj_resetData(model, data)
            targets = state.snapshot()
            data.ctrl[ids] = [targets[k] for k in names]
            mujoco.mj_step(model, data, nstep=5)
            ticks += 5
            if data.warning.number.any() or not np.isfinite(data.qpos).all():
                raise RuntimeError(f"Simulation warning at {data.time}s: {data.warning.number}")
            state.update(angles=dict(zip(names, data.qpos[qadr].tolist())), time=float(data.time))
            if viewer:
                viewer.sync()
                time.sleep(max(0, 5*model.opt.timestep-(time.monotonic()-start)))
        print(json.dumps({"scene": SCENE.name, "seconds": data.time, "contacts": data.ncon, "actuators": model.nu}))
