"""Compare directed wheel steering, tire slip and whole-robot COM at 500 Hz."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import numpy as np
import mujoco as mj

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'grounded_phi_buffer'))
from control import prepare,target
import controllers.grounded_morph as morph


def run(e,grid,path,travel,mode):
    m,d=e.model,e.data
    floor=m.geom('ctm_floor').id
    tires={m.pair_geom2[m.pair(f'rubber_floor_w{i+1}').id]:i for i in range(4)}
    mj.mj_forward(m,d)
    start=d.subtree_com[e.base_body].copy()
    previous=path[0].copy()
    steer0=e.joint_positions(e.steering_ids)
    demand=np.zeros(2)
    rows=[]
    steer_and_roll=morph._steer_and_roll
    def instrument(env,v,**options):
        nonlocal request
        request=v.copy()
        if mode=='coast':
            actuators=np.r_[env.steering_actuators,env.wheel_actuators]
            d.ctrl[actuators]=d.qpos[m.jnt_qposadr[np.r_[env.steering_ids,env.wheel_ids]]]
        else:
            steer_and_roll(env,v,**options)
    e.control_steps=1
    e._morph_motion=None
    with patch.object(morph,'_steer_and_roll',instrument):
        for k in range(1,round(6/e.dt)+1):
            t=k*e.dt
            phi=morph.moving_phi(t)
            q=target(phi,grid,path)
            twist=travel*min(1,t/.5,(6-t)/.5)
            demand+=twist[:2]*e.dt
            request=np.zeros((4,2))
            e.drive_morph(q,(q-previous)/e.dt,*twist)
            previous=q
            load=np.zeros(4)
            slip=np.zeros(4)
            for i,c in enumerate(d.contact):
                if floor not in c.geom: continue
                geom=c.geom2 if c.geom1==floor else c.geom1
                if geom not in tires: continue
                wrench=np.empty(6)
                mj.mj_contactForce(m,d,i,wrench)
                normal=max(0.,wrench[0])
                jac=np.empty((3,m.nv))
                mj.mj_jac(m,d,jac,None,c.pos,m.geom_bodyid[geom])
                # Tangential velocity of the tire material at the contact with the static floor.
                vt=c.frame.reshape(3,3)[1:] @ (jac @ d.qvel)
                n=tires[geom]
                load[n]+=normal
                slip[n]+=normal*(vt@vt)
            pos=d.subtree_com[e.base_body].copy()-start
            rows.append((t,phi,pos,demand.copy(),load,slip,e.joint_positions(e.steering_ids),
                         d.ctrl[e.steering_actuators].copy(),request,d.qpos.copy(),d.xaxis[e.wheel_ids].copy()))
    assert not d.warning.number.any()
    z=dict(zip(('t','phi','com','demand','load','slip_energy','steer','ctrl','request','qpos','axes'),map(np.array,zip(*rows))))
    err=1000*np.linalg.norm(z['com'][:,:2]-z['demand'],axis=1)
    total=np.degrees(np.abs(np.diff(z['steer'],axis=0,prepend=steer0[None,:])).sum(axis=0))
    loaded=z['load']>.1
    w=np.where(loaded,z['load'],0)
    energy=np.where(loaded,z['slip_energy'],0)
    summary=dict(max_err_mm=err.max(),end_mm=err[-1],range_z_mm=np.ptp(z['com'][:,2])*1000,
                 steer_travel_deg=total.round(1).tolist(),slip_rms_mm_s=1000*np.sqrt(energy.sum()/w.sum()),
                 slip_per_wheel_mm_s=(1000*np.sqrt(energy.sum(0)/w.sum(0))).round(2).tolist(),
                 four_pct=100*loaded.all(1).mean())
    return summary,z


if __name__=='__main__':
    e,g,p=prepare()
    snap=mj.MjData(e.model)
    mj.mj_copyData(snap,e.model,e.data)
    label=sys.argv[1]
    for name,travel,mode in [('coast',[0,0,0],'coast'),('still',[0,0,0],'active'),('moving',[.04,.03,.12],'active')]:
        mj.mj_copyData(e.data,e.model,snap)
        metrics,z=run(e,g,p,np.array(travel),mode)
        print(name,metrics,flush=True)
        np.savez_compressed(OUT/(label+'_'+name+'.npz'),**z)
        (OUT/(label+'_'+name+'.json')).write_text(json.dumps(metrics,indent=2))
