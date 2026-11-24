"""Grounded CAD sign mapping, constrained references and nominal servo test.

Run with PYTHONPATH=src from the repository root. All caps come from scene.xml.
"""
from pathlib import Path
import csv
import json
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from controllers.mujoco_control import angles_from_phi
from envs.tars_grounded import SCENE, TarsGroundedEnv

OUT = Path(__file__).resolve().parent
OLD = ('joint1','joint2','joint3','joint_closure','joint7','joint6','joint5','joint4')


def mapping(env):
    old = mujoco.MjModel.from_xml_path(str(SCENE.parent.parent / 'robot_closed_loop.xml'))
    old_data, new_data = mujoco.MjData(old), mujoco.MjData(env.model)
    mujoco.mj_forward(old, old_data)
    mujoco.mj_forward(env.model, new_data)
    ids = [old.joint(name).id for name in OLD]
    x, y = old_data.xanchor[ids,:2], new_data.xanchor[env.block_ids,:2]
    x, y = x-x.mean(0), y-y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = np.eye(3)
    rotation[:2,:2] = u @ vt
    dots = np.sum((old_data.xaxis[ids] @ rotation)*new_data.xaxis[env.block_ids],axis=1)
    assert np.linalg.det(rotation) > .999 and min(abs(dots)) > .99
    # Along the new serial chain, old J7, J6, J5 run child-to-parent.
    return np.sign(dots)*[1,1,1,1,-1,-1,-1,1], dots


def weld(data, sites):
    a,b = sites
    ra,rb = data.site_xmat[sites].reshape(2,3,3)
    return data.site_xpos[b]-data.site_xpos[a], Rotation.from_matrix(ra.T@rb).as_rotvec()


def path_for(env, signs):
    model = env.model
    kin = mujoco.MjData(model)
    sites = [model.eq_obj1id[0],model.eq_obj2id[0]]
    adr = model.jnt_qposadr[env.block_ids]
    lo,hi = model.jnt_range[env.block_ids].T
    def residual(q):
        kin.qpos[adr] = q
        mujoco.mj_kinematics(model,kin)
        p,r = weld(kin,sites)
        return np.r_[p/.2,r]
    assert np.linalg.norm(residual(np.zeros(8))) < 1e-10
    phis = np.linspace(0,3,601)
    raw = []
    for phi in phis:
        j = angles_from_phi(phi)
        raw.append(signs * np.r_[j[0:3],-j[3],j[6],j[5],j[4],j[3]])
    raw = np.array(raw)
    path = np.zeros_like(raw)
    mid = len(phis)//2
    for idx in (range(mid-1,-1,-1),range(mid+1,len(phis))):
        prev = path[mid]
        for i in idx:
            fit = least_squares(lambda q: np.r_[residual(q),.001*(q-raw[i])],
                                np.clip(prev+raw[i]-raw[i-(1 if i>mid else -1)],lo+1e-9,hi-1e-9),
                                bounds=(lo,hi),ftol=1e-12,xtol=1e-12,gtol=1e-12)
            path[i] = prev = fit.x
    errors = np.array([residual(q) for q in path])
    assert np.max(np.linalg.norm(errors,axis=1)) < 1e-5
    return phis,path,float(np.degrees(abs(path-raw)).max())


def run(rate=.5, movie=False):
    env = TarsGroundedEnv()
    m,d = env.model,env.data
    signs,dots = mapping(env)
    phis,path,correction = path_for(env,signs)
    act = env.block_actuators
    adr = m.jnt_qposadr[env.block_ids]
    caps = m.actuator_forcerange[act,1]
    assert np.all(caps == 11) and np.all(m.actuator_gear[act,0] == 1)
    assert np.all(m.actuator_forcerange[env.wheel_actuators,1] == .45)
    sites = [m.eq_obj1id[0],m.eq_obj2id[0]]
    floor = m.geom('ctm_floor').id
    pairs = [m.pair(f'rubber_floor_w{i}').id for i in range(1,5)]
    tires = set(m.pair_geom2[pairs])
    assert len(tires)==4 and np.all(m.pair_friction[pairs,:2] == .8)
    balls = set(m.pair_geom2)-tires
    # Drive and steering hold their reset position targets, with stock force caps.
    def tick(phi):
        d.ctrl[act] = [np.interp(phi,phis,col) for col in path.T]
        mujoco.mj_step(m,d)
    for phi in np.linspace(1.5,0,round(1.5/rate/env.dt)+1)[1:]:
        tick(phi)
    for _ in range(round(2/env.dt)):
        tick(0)
    camera = mujoco.MjvCamera()
    camera.distance,camera.elevation,camera.azimuth = 1.5,-23,120
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    renderer = mujoco.Renderer(m,360,600) if movie else None
    frames, rows, loads = [],[],[]
    steps = round(3/rate/env.dt)
    count = {g:0.0 for g in range(m.ngeom)}
    for k in range(steps):
        phi = (k+1)*3/steps
        tick(phi)
        # Force and target before integration belong to the same physics step.
        torque = d.actuator_force[act].copy()
        mujoco.mj_forward(m,d)
        p,r = weld(d,sites)
        support = np.zeros(3)
        active = set()
        for ci,c in enumerate(d.contact):
            if floor not in c.geom: continue
            g = int(c.geom2 if c.geom1 == floor else c.geom1)
            wrench = np.empty(6)
            mujoco.mj_contactForce(m,d,ci,wrench)
            normal = max(0,wrench[0])
            if g in tires and normal > .1: active.add(g)
            support[0 if g in tires else 1 if g in balls else 2] += normal
            if phi>2.5: count[g] += normal*env.dt
        loads.append(support)
        rows.append(np.r_[phi,torque,d.qpos[adr],d.ctrl[act],np.linalg.norm(p)*1000,
                           np.degrees(np.linalg.norm(r)),len(active)])
        if movie and (k%25==0 or k==steps-1):
            camera.lookat[:] = d.subtree_com[0]
            renderer.update_scene(d,camera=camera,scene_option=opt)
            frame = Image.fromarray(renderer.render())
            ImageDraw.Draw(frame).text((12,12),f'phi {phi:.2f}    loaded tires {len(active)}/4',fill='white',stroke_width=2,stroke_fill='black')
            frames.append(frame.convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
    if renderer:
        renderer.close()
        frames[0].save(OUT/'cad_phi.gif',save_all=True,append_images=frames[1:],duration=50,loop=0)
    rows,loads = np.array(rows),np.array(loads)
    assert not d.warning.number.any() and np.isfinite(rows).all()
    f = abs(rows[:,1:9])
    error = np.degrees(abs(rows[:,9:17]-rows[:,17:25]))
    prefix = 'cad' if rate==.5 else 'cad_slow'
    np.savez_compressed(OUT/f'{prefix}_signals.npz',phi=rows[:,0],torque_nm=rows[:,1:9],
                        actual_rad=rows[:,9:17],target_rad=rows[:,17:25],gap_mm=rows[:,-3],
                        weld_deg=rows[:,-2],loaded_tires=rows[:,-1],support_N=loads)
    joint_rows = []
    for i in range(8):
        peak = f[:,i].argmax()
        joint_rows.append([i+1,np.sqrt(np.mean(f[:,i]**2)),f[peak,i],rows[peak,0],
                           np.count_nonzero(f[:,i]>=caps[i]-1e-7)*env.dt,error[:,i].max()])
    with (OUT/f'{prefix}_joints.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dof','rms_Nm','peak_Nm','phi_at_peak','at_XML_cap_s','max_error_deg'])
        writer.writerows(joint_rows)
    print(prefix,'sign',signs,'axis_dot',dots,'max correction deg',correction,
          'weld mm/deg',rows[:,-3:-1].max(0),'tracking max/end deg',error.max(),error[-1].max(),
          'four tire fraction',np.mean(rows[:,-1]==4),'late loads',loads[rows[:,0]>2.5].mean(0),
          'late top geoms',[(m.geom(g).name,round(count[g],2)) for g in sorted(count,key=count.get,reverse=True)[:4]],
          'joints',np.array(joint_rows).round(3), flush=True)
    return rows


if __name__=='__main__':
    rows = run(movie=True)
    run(.1)
    import matplotlib.pyplot as plt
    fig,(ax,ct) = plt.subplots(2,1,sharex=True,figsize=(9.5,5.8),height_ratios=[3,1],layout='constrained')
    for i in range(8):
        ax.plot(rows[:,0],rows[:,i+1],lw=1,label=f'J{i+1}')
    for bound in [-11,11]: ax.axhline(bound,color='#a43926',lw=1)
    ax.legend(ncol=8,loc='upper center',frameon=False)
    ax.set(ylabel='Joint torque (N m)',ylim=(-12.5,14),title='CAD-mapped eight-axis sweep | 6 s | XML limit +/-11 N m')
    ct.plot(rows[:,0],rows[:,-1],lw=1,color='#33645c')
    ct.set(xlabel='Commanded phi',ylabel='Loaded tires',yticks=range(5),xlim=(0,3))
    for a in (ax,ct): a.grid(False,which='both')
    fig.savefig(OUT/'cad_torques.png',dpi=160)
