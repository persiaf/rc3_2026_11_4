"""Read-only frozen display data and common publication plotting primitives."""
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
INDEX=json.loads((HERE/'sources/display_manifest.json').read_text(encoding='utf-8'))
LOCAL=json.loads((HERE/'sources/source_manifest.json').read_text(encoding='utf-8'))
COLORS={'route':'#254B73','L':'#0072B2','R':'#D55E00','table':'#E7E7E7',
        'edge':'#858585','obstacle':'#BDBDBD','candidate':'#A3A3A3','depot':'#111111'}
MARKERS={'L':'^','R':'s'}
plt.rcParams.update({'font.family':['Times New Roman','SimSun','DejaVu Serif'],'font.serif':['Times New Roman','SimSun','DejaVu Serif'],
    'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':9,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.05,
    'axes.grid':False,'axes.spines.top':False,'axes.spines.right':False,'text.usetex':False,
    'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'lines.solid_capstyle':'round'})


def digest(data):return hashlib.sha256(data).hexdigest()


def source(name):
    record=LOCAL['sources'][name];data=(HERE/record['local_path']).read_bytes()
    if digest(data)!=record['sha256']:raise ValueError('source hash mismatch: '+name)
    return json.loads(data.decode('utf-8'))


def pointer(document,path):
    value=document
    if not path:return value
    for token in path.lstrip('/').split('/'):
        token=token.replace('~1','/').replace('~0','~')
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value


def resolve(ref):return pointer(source(ref['source']),ref['json_pointer'])


def display(display_id):return next(d for d in INDEX['displays'] if d['id']==display_id)


def scene_bounds(scene,pad=.4):
    xs=[float(scene['start'][0])];ys=[float(scene['start'][1])]
    for t in scene['tables']:xs.extend((t['xmin'],t['xmax']));ys.extend((t['ymin'],t['ymax']))
    for c in scene['candidates']:xs.append(c['x']);ys.append(c['y'])
    xs.extend(scene.get('x_lines',[]));ys.extend(scene.get('y_lines',[]))
    return (min(xs)-pad,max(xs)+pad,min(ys)-pad,max(ys)+pad)


def draw_scene(ax,scene,bounds=None,table_ids=False,candidates=False):
    for t in scene['tables']:
        ax.add_patch(patches.Rectangle((t['xmin'],t['ymin']),t['xmax']-t['xmin'],t['ymax']-t['ymin'],
                                      facecolor=COLORS['table'],edgecolor=COLORS['edge'],lw=.6,zorder=1))
        if table_ids:ax.text(t['cx'],t['cy'],str(t['id']),color='#696969',ha='center',va='center',fontsize=8,zorder=2)
    for o in scene.get('obstacles',[]):
        ax.add_patch(patches.Rectangle((o['xmin'],o['ymin']),o['xmax']-o['xmin'],o['ymax']-o['ymin'],
                                      facecolor=COLORS['obstacle'],edgecolor='#666666',hatch='////',lw=.6,zorder=2))
    if candidates:
        ax.scatter([c['x'] for c in scene['candidates']],[c['y'] for c in scene['candidates']],
                   marker='.',s=8,c=COLORS['candidate'],zorder=2)
    b=bounds or scene_bounds(scene)
    ax.set_xlim(b[:2]);ax.set_ylim(b[2:]);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('$x$ / m');ax.set_ylabel('$y$ / m')
    ax.tick_params(direction='out',length=3,width=.6)


def draw_route(ax,plan,lw=1.3,color=None,arrows=False):
    xy=np.asarray(plan['route_polyline'],float)
    ax.plot(xy[:,0],xy[:,1],color=color or COLORS['route'],lw=lw,zorder=3)
    if arrows:
        lengths=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        cumulative=np.concatenate(([0.],np.cumsum(lengths)))
        # Pure display annotations along stored segments; never interpolate a new path.
        for target in np.linspace(0,cumulative[-1],10)[1:-1]:
            i=min(int(np.searchsorted(cumulative,target)-1),len(lengths)-1)
            if lengths[i]<.22:continue
            mid=(xy[i]+xy[i+1])/2;u=(xy[i+1]-xy[i])/lengths[i]
            ax.annotate('',xy=mid+.12*u,xytext=mid-.12*u,
                        arrowprops={'arrowstyle':'->','color':color or COLORS['route'],'lw':lw},zorder=4)


def draw_tasks(ax,scene,plan=None,size=20,labels=False):
    assigned={a['task_id']:a['arm'] for a in plan['assignments']} if plan else {}
    for arm in (None,'L','R'):
        tasks=[t for t in scene['tasks'] if assigned.get(t['id'])==arm]
        if not tasks:continue
        ax.scatter([t['x'] for t in tasks],[t['y'] for t in tasks],s=size,
                   marker=MARKERS.get(arm,'o'),facecolor=COLORS.get(arm,'#202020'),
                   edgecolor='white',lw=.35,zorder=7)
        if labels:
            for t in tasks:
                ax.annotate(f"{t['id']}{arm or ''}",(t['x'],t['y']),xytext=(4,4),textcoords='offset points',
                            fontsize=8,color=COLORS.get(arm,'#202020'),zorder=8)


def draw_stops(ax,scene,plan,size=30,numbered=False,headings=False):
    for stop in plan['service_stops']:
        p=stop['pose'];ax.scatter([p['x']],[p['y']],s=size,facecolor='white',edgecolor='#111111',lw=.8,zorder=6)
        if numbered:
            ax.text(p['x'],p['y'],str(stop['sequence']),ha='center',va='center',fontsize=8,zorder=8)
        if headings:
            ax.arrow(p['x'],p['y'],.25*math.cos(p['theta']),.25*math.sin(p['theta']),
                     width=.012,head_width=.08,length_includes_head=True,color='#222222',zorder=5)
    ax.scatter([scene['start'][0]],[scene['start'][1]],marker='*',s=75,color=COLORS['depot'],zorder=9)


def legend_handles(include_candidates=False,include_obstacles=False):
    handles=[patches.Patch(facecolor=COLORS['table'],edgecolor=COLORS['edge'],label='桌面'),
             Line2D([],[],color=COLORS['route'],lw=1.4,label='底盘路径'),
             Line2D([],[],ls='',marker='o',mfc='white',mec='#111111',label='服务停靠'),
             Line2D([],[],ls='',marker='*',color=COLORS['depot'],markersize=9,label='起终点'),
             Line2D([],[],ls='',marker='^',color=COLORS['L'],label='左臂任务 L'),
             Line2D([],[],ls='',marker='s',color=COLORS['R'],label='右臂任务 R')]
    if include_candidates:handles.append(Line2D([],[],ls='',marker='.',color=COLORS['candidate'],label='未选候选'))
    if include_obstacles:handles.append(patches.Patch(facecolor=COLORS['obstacle'],hatch='////',label='附加障碍'))
    return handles


def cost_summary(plan):
    xy=plan['route_polyline'];length=sum(math.dist(a,b) for a,b in zip(xy,xy[1:]))
    stops=len(plan['selected_candidate_ids']);penalty=plan['stop_penalty_equivalent_m'];cost=length+penalty*stops
    if abs(length-plan['route_length_m'])>1e-7 or abs(cost-plan['objective_length_plus_stops_m'])>1e-7:
        raise ValueError('saved route cost inconsistent')
    return {'L_m':length,'K':stops,'lambda_m':penalty,'J_equivalent_m':cost}


def save(fig,name):
    result={}
    for fmt in ('pdf','svg','png'):
        path=HERE/f'{name}.{fmt}';fig.savefig(path,format=fmt);result[fmt]=path.name
    return result


def write_json(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
