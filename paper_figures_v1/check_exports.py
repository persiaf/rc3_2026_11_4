"""Refresh file-level export checks after rendering; visual review is separate."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
from PIL import Image

HERE=Path(__file__).resolve().parent


def main():
    figures=[]
    for stem in ('fig1_scene','fig2_constraints','fig3_routes100','fig3_zoom'):
        outputs={}
        for ext in ('pdf','svg','png'):
            path=HERE/f'{stem}.{ext}';data=path.read_bytes();assert data
            record={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if ext=='pdf':
                record.update(image_xobjects=data.count(b'/Subtype /Image'),embedded_truetype_fonts=data.count(b'/FontFile2'))
                assert record['image_xobjects']==0 and record['embedded_truetype_fonts']>0
            if ext=='svg':
                record['editable_text_count']=sum(e.tag.endswith('}text') for e in ET.fromstring(data).iter())
                assert record['editable_text_count']>0
            if ext=='png':
                with Image.open(path) as im:record.update(size=list(im.size),dpi=list(im.info['dpi']))
                assert all(abs(d-300)<.01 for d in record['dpi'])
            outputs[ext]=record
        figures.append({'stem':stem,'outputs':outputs})
    report={'all_nonempty':True,'all_passed':True,'figures':figures,
            'scope':'File-level checks after current rendering; see FIGURE_REVIEW.md for separate human-readable visual review.'}
    (HERE/'export_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Refresh the earlier fig3-specific receipt instead of retaining stale PDF/SVG hashes.
    limited=dict(report,figures=figures[2:])
    (HERE/'fig3_output_check.json').write_text(json.dumps(limited,indent=2),encoding='utf-8')
    print(json.dumps({'all_passed':True,'figures':len(figures),'formats_per_figure':3}))


if __name__=='__main__':main()
