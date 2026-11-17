"""Exact planar3R theta3=0 checkpoint slices, reliability splits and feasible path."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog, root, minimize
import networkx as nx
from scripts.planar3R_iris_regions import load_iris_checkpoint, interior_point
from scripts.planar3R_bbox_reliability_partition import (
    branch_bounding_box, build_reliability_lookup, build_color_lookup,
    partition_single_branch, R1,R2,R3)
from scripts.planar3R_collision import is_collision_free

out=Path(__file__).resolve().parent
regs,meta=load_iris_checkpoint('data/iris/planar3R_iris_regions.npz')
ssm=np.load(out/'goal_ssm_data.npz')['ssm_0']
bounds=branch_bounding_box(ssm)
lookup=build_reliability_lookup(R1,R2,R3)
colors,sm=build_color_lookup(lookup)
parts=[c for c in partition_single_branch(bounds,lookup) if c['lower'][2]<=0<c['upper'][2]]
def poly(A,b):
    c=interior_point(A,b)
    if c is None:return None
    v=HalfspaceIntersection(np.c_[A,-b],c).intersections
    return v[ConvexHull(v).vertices]
slices=[]
for i,r in enumerate(regs):
    A=r.A[:,:2]; b=r.b.copy()
    v=poly(A,b)
    if v is not None:slices.append(dict(A=A,b=b,v=v,parent=i))
cells=[]
for s in slices:
    for part in parts:
        A=np.vstack([s['A'],np.eye(2),-np.eye(2)])
        b=np.r_[s['b'],part['upper'][:2],-part['lower'][:2]]
        v=poly(A,b)
        if v is not None:cells.append(dict(A=A,b=b,v=v,R=part['reliability'],parent=s['parent']))
start=np.array([-2.972349782093813,-2.26926174311021])
def fk(q):
    angles=np.cumsum([*q,0.])
    return np.array([np.cos(angles).sum(),np.sin(angles).sum()])
goal=root(lambda q:fk(q)-[1.5,1.5],[-.4254,1.6975]).x
def contains(s,q):return np.all(s['A']@q<=s['b']+1e-7)
G=nx.Graph();G.add_nodes_from(range(len(cells)))
portals={}
for i,s in enumerate(cells):
    for j in range(i+1,len(cells)):
        t=cells[j]
        lp=linprog([0,0],A_ub=np.vstack([s['A'],t['A']]),b_ub=np.r_[s['b'],t['b']],bounds=[(None,None)]*2,method='highs')
        if lp.success:
            G.add_edge(i,j,weight=np.linalg.norm(s['v'].mean(0)-t['v'].mean(0)) / max(.05,(s['R']+t['R'])/2))
            portals[i,j]=portals[j,i]=lp.x
starts=[i for i,c in enumerate(cells) if contains(c,start)]
goals=[i for i,c in enumerate(cells) if contains(c,goal)]
routes=[]
for i in starts:
    for j in goals:
        if nx.has_path(G,i,j):
            r=nx.shortest_path(G,i,j,weight='weight')
            routes.append((sum(G[a][b]['weight'] for a,b in zip(r[:-1],r[1:])),r))
if not routes:raise RuntimeError('No connected slice route')
route=min(routes,key=lambda x:x[0])[1]
x0=np.array([portals[a,b] for a,b in zip(route[:-1],route[1:])])
def unpack(z):return np.vstack([start,z.reshape(-1,2),goal])
weights=np.array([1/max(cells[i]['R'],.05) for i in route])
def cost(z):return np.sum(np.linalg.norm(np.diff(unpack(z),axis=0),axis=1)*weights)
def constraint(z):
    points=z.reshape(-1,2);result=[]
    for k,p in enumerate(points):
        for idx in [route[k],route[k+1]]:
            c=cells[idx];result.extend(c['b']-c['A']@p)
    return np.array(result)
res=minimize(cost,x0.flatten(),method='SLSQP',constraints=[dict(type='ineq',fun=constraint)],options={'maxiter':1000,'ftol':1e-10})
if not res.success or np.min(constraint(res.x)) < -1e-6:raise RuntimeError(str(res.message))
path=unpack(res.x)
for k,i in enumerate(route):
    assert contains(cells[i],path[k]) and contains(cells[i],path[k+1])
samples=np.vstack([(1-t)*a+t*b for a,b in zip(path[:-1],path[1:]) for t in np.linspace(0,1,80)])
assert all(is_collision_free(np.r_[p,0.],meta['model']['obstacles'],link_radius=.02) for p in samples)
assert np.linalg.norm(fk(goal)-[1.5,1.5])<1e-7

# Collision geometry for the same plane; dark fill means sampled collision.
grid=np.linspace(-np.pi,np.pi,241); X,Y=np.meshgrid(grid,grid)
angle=np.stack([X,X+Y,X+Y])
px=np.concatenate([np.zeros((1,*X.shape)),np.cumsum(np.cos(angle),axis=0)])
py=np.concatenate([np.zeros((1,*X.shape)),np.cumsum(np.sin(angle),axis=0)])
collision=np.zeros_like(X,dtype=bool)
for k in range(3):
    dx=px[k+1]-px[k];dy=py[k+1]-py[k]
    t=np.clip(((.75-px[k])*dx+(.75-py[k])*dy)/(dx*dx+dy*dy),0,1)
    collision|=(px[k]+t*dx-.75)**2+(py[k]+t*dy-.75)**2 <= .17**2

plt.rcParams.update({'font.family':'DejaVu Serif','svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for k,ax in enumerate(axes):
    ax.set_title('')
    ax.set_facecolor('#FAFCFF')
    ax.contourf(grid,grid,collision.astype(int),levels=[.5,1.5],colors=['#B5B5B5'])
    collection=slices if k==0 else cells
    for c in collection:
        ax.add_patch(Polygon(c['v'],facecolor='#DFEFFF' if k==0 else colors[c['R']],
                             edgecolor='#2662BD' if k==0 else '#527346',lw=1,alpha=.9))
    if k==2:
        ax.plot(*path.T,color='#8800EE',lw=3,zorder=7)
        ax.scatter(*start,s=55,c='#1BEB17',edgecolor='black',zorder=8)
        ax.scatter(*goal,s=55,c='#FA2525',edgecolor='black',zorder=8)
    ax.set(xlim=(-np.pi,np.pi),ylim=(-np.pi,np.pi),xlabel=r'$q_1$',ylabel=r'$q_2$')
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(['1. Collision-free regions','3. Exact intersections','4. Optimized slice trajectory'][k],fontsize=13)
fig.subplots_adjust(left=.04,right=.99,bottom=.14,top=.88,wspace=.22)
fig.savefig(out/'framework_data_panels.png',dpi=250)
fig.savefig(out/'framework_data_panels.svg')
for ax in axes:
    ax.set_title('')
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
fig.canvas.draw()
for k,ax in enumerate(axes):
    extent=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.28,1.28)
    fig.savefig(out/f'framework_panel_{[1,3,4][k]}.png',dpi=300,bbox_inches=extent)
data=dict(theta3=0,source_checkpoint='data/iris/planar3R_iris_regions.npz',
          source_regions=len(slices),intersection_regions=len(cells),goal_position=[1.5,1.5],
          start_configuration=[*start,0],goal_configuration=[*goal,0],path=np.c_[path,np.zeros(len(path))].tolist(),
          note='New trajectory optimized in the displayed theta3=0 slice, not the original varying-theta3 experiment path.',
          collision_checked_samples=len(samples),objective=float(res.fun))
(out/'framework_panel_data.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
