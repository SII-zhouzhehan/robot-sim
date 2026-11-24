"""Sweep the CAD reference with bounded wheel coordination and stop-to-stop timing."""
from pathlib import Path
import sys
import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'experiments/grounded_phi0_3')]
from cad_sweep import mapping, path_for, weld
from envs.tars_grounded import TarsGroundedEnv
from controllers.grounded_morph import buffered_phi, coordinate_morph_wheels

OUT = Path(__file__).resolve().parent


def target(p, grid, path):
    return np.array([np.interp(p,grid,col) for col in path.T])


def prepare():
    e=TarsGroundedEnv()
    grid,path,_=path_for(e,mapping(e)[0])
    for p in np.linspace(1.5,0,round(3/e.dt)+1)[1:]:
        e.data.ctrl[e.block_actuators]=target(p,grid,path)
        mj.mj_step(e.model,e.data)
    for _ in range(round(2/e.dt)):
        mj.mj_step(e.model,e.data)
    return e,grid,path


def run(e,grid,path,mode,duration,ramp):
    m,d=e.model,e.data
    saved=mj.MjData(m)
    mj.mj_copyData(saved,m,d)
    act=e.block_actuators
    pairs=[m.pair(f'rubber_floor_w{i}').id for i in range(1,5)]
    tires=m.pair_geom2[pairs]
    floor=m.geom('ctm_floor').id
    sites=[m.eq_obj1id[0],m.eq_obj2id[0]]
    last=path[0].copy()
    rows=[]
    for k in range(1,round(duration/e.dt)+1):
        p=buffered_phi(k*e.dt,duration,ramp)
        ref=target(p,grid,path)
        d.ctrl[act]=ref
        if mode=='coast':
            # Zero position error at this physics step (zero current request).
            d.ctrl[e.wheel_actuators]=e.joint_positions(e.wheel_ids)
        elif mode=='follow':
            coordinate_morph_wheels(e,(ref-last)/e.dt)
        last=ref
        mj.mj_step(m,d)
        forces=d.actuator_force[act].copy()
        mj.mj_forward(m,d)
        normal=np.zeros(4)
        for ci,c in enumerate(d.contact):
            if floor in c.geom:
                other=c.geom2 if floor==c.geom1 else c.geom1
                idx=np.flatnonzero(tires==other)
                if len(idx):
                    w=np.empty(6)
                    mj.mj_contactForce(m,d,ci,w)
                    normal[idx[0]]+=max(0,w[0])
        twist=np.empty(6)
        mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,e.base_body,twist,0)
        dp,dr=weld(d,sites)
        rows.append((p,d.subtree_com[e.base_body,2],np.rad2deg(np.linalg.norm(twist[:2])),normal,forces,d.qpos.copy(),ref,
                     np.linalg.norm(dp)*1000,np.rad2deg(np.linalg.norm(dr))))
    if d.warning.number.any(): raise RuntimeError(d.warning.number.copy())
    mj.mj_copyData(d,m,saved)
    return dict(zip(('phi','z','angular','normal','torque','qpos','ref','gap','angle'),map(np.asarray,zip(*rows))))


def metrics(e,r):
    p=r['phi']; az=np.gradient(np.gradient(r['z'],e.dt),e.dt)
    loaded=(r['normal']>.1).sum(1)
    report={}
    for key,mask in [('all',p>=0),('one',(p>=1-1e-10)&(p<=1.12+1e-10)),('two',(p>=2-1e-10)&(p<=2.12+1e-10))]:
        report[key]=np.round([np.sqrt(np.mean(az[mask]**2)),r['angular'][mask].max(),100*np.mean(loaded[mask]==4),abs(r['torque'][mask]).max(),r['normal'][mask].max(),loaded[mask].min()],4).tolist()
    return report

if __name__=='__main__':
    e,grid,path=prepare()
    result=run(e,grid,path,'follow',6.0,.45)
    print(metrics(e,result))
