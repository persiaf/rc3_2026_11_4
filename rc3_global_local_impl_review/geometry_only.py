"""Frozen-generator-compatible physical layouts, without any target task draw.

The mixed-mode layout RNG consumes the original density/count *offset* because
it affects subsequent geometry draws. No task coordinates, service labels,
target task seeds, F, navigation or model are materialized here.
"""
import numpy as np
from rc3_baseline.baseline import make_scene
from rc3_fast.planner import map_key


def physical_payload(scene):
    p={k:scene[k] for k in ('tables','candidates','start','x_lines','y_lines','robot')}
    p['obstacles']=scene.get('obstacles',[])
    return p


def make_layout(nx,ny,seed,mode):
    if mode not in ('regular','mixed','blocked'):raise ValueError('Unknown layout mode')
    if any(type(x) is not int for x in (nx,ny,seed)) or nx<1 or ny<1:raise ValueError('Invalid layout declaration')
    if mode=='regular':
        scene=make_scene(nx,ny,seed,counts=[0]*(nx*ny))
        scene.update(complexity=mode,obstacles=[])
        assert not scene['tasks']
        return scene
    rng=np.random.default_rng(seed+7919)
    density=int(rng.integers(3))
    probabilities=((.45,.30,.20,.05),(.25,.25,.25,.25),(.05,.10,.30,.55))[density]
    rng.choice(4,size=nx*ny,p=probabilities)  # discarded; layout-RNG compatibility only
    geometry=dict(table_width=float(rng.uniform(.85,.95)),table_depth=float(rng.uniform(.60,.75)),aisle_gap=float(rng.uniform(1.15,1.7)))
    scene=make_scene(nx,ny,seed,counts=[0]*(nx*ny),geometry=geometry)
    for table in scene['tables']:
        old_x,old_y=table['cx'],table['cy'];sx,sy=rng.uniform(.95,1.05,2);dx,dy=rng.uniform(-.04,.04,2)
        table['cx'],table['cy']=old_x+dx,old_y+dy
        for low,high,center,shift,scale in (('xmin','xmax',old_x,dx,sx),('ymin','ymax',old_y,dy,sy)):
            table[low]=center+shift+(table[low]-center)*scale
            table[high]=center+shift+(table[high]-center)*scale
        width,depth=table['xmax']-table['xmin'],table['ymax']-table['ymin']
        for local,c in enumerate(scene['candidates'][table['id']*8:table['id']*8+8]):
            frac=-.25 if local%2==0 else .25;side=c['side']
            if side in ('bottom','top'):
                c['x']=table['cx']+frac*width;c['y']=table['cy']+(-1 if side=='bottom' else 1)*(depth/2+.5)
                c['corridor_anchor'][0]=c['x']
            else:
                c['x']=table['cx']+(-1 if side=='left' else 1)*(width/2+.5);c['y']=table['cy']+frac*depth
                c['corridor_anchor'][1]=c['y']
    obstacles=[]
    if mode=='blocked':
        slots=[(ix,iy) for iy in range(1,ny,2) for ix in range(nx)]
        if slots:
            selected=rng.choice(len(slots),size=min(len(slots),max(1,round(nx*ny*.08))),replace=False)
            for index in selected:
                ix,iy=slots[index];x=(scene['x_lines'][ix]+scene['x_lines'][ix+1])/2;y=scene['y_lines'][iy]
                obstacles.append(dict(xmin=x-.06,xmax=x+.06,ymin=y-.20,ymax=y+.20))
    scene.update(complexity=mode,density_profile=density,obstacles=obstacles)
    assert scene['tasks']==[]
    return scene
