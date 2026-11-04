"""Common fixed central crop of the same five stored routes (no re-planning)."""
from gen_fig3_routes100 import *


def main():
    item,scene,rows,bounds,crop=frozen_case()
    fig=panel_layout(scene,rows,crop,crop,zoom=True)
    outputs=save(fig,'fig3_zoom');plt.close(fig)
    write_json('fig3_zoom_metadata.json',{'seed':scene['seed'],'source':INDEX['sources']['main100'],
        'common_axes_limits':list(crop),'crop_rule':'Central 20% of the saved navigation-coordinate envelope on each axis.',
        'task_ids_in_crop':[t['id'] for t in scene['tasks'] if crop[0]<=t['x']<=crop[1] and crop[2]<=t['y']<=crop[3]],
        'methods':[r['label'] for r in rows],'outputs':outputs,'new_planning_or_inference':False,
        'scope':'Spatial detail only. The identical crop is fixed independently of which method is better; clipped lines continue outside the view.'})


if __name__=='__main__':main()
