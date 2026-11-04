"""Bundle code, frozen weights, manifests and results; omit derived multi-GB caches."""
from pathlib import Path
import zipfile

def main():
    root=Path.cwd();target=root/'rc3_improve_delivery_v2.zip'
    files=set()
    for name in ('rc3_baseline','rc3_fast','rc3_improve'):
        files.update((root/name).glob('*.py'))
    for pattern in ('*.md','requirements.txt'):
        files.update((root/'rc3_improve').glob(pattern))
    for directory in ('deployment_v2','run_v2_20261102','results_v2_20261102','verification_v2','figures'):
        files.update(p for p in (root/'rc3_improve'/directory).rglob('*') if p.is_file())
    files.update((root/'rc3_improve/data_v2_20261102').glob('*.json'))
    files.update((root/'rc3_improve').glob('*_v2.json'))
    files.add(root/'rc3_improve/SPLIT_AUDIT.json')
    files.add(root/'rc3_fast/deployment/selector.npz')
    files.add(root/'rc3_review_import/IMPORT_AUDIT.md')
    start='''# 使用入口

最终模型：rc3_improve/deployment_v2/policy.pt
方案与运行命令：rc3_improve/README.md
最终实验：rc3_improve/RESULTS.md
原始测试：rc3_improve/results_v2_20261102/
独立复验：rc3_improve/verification_v2/verification.json

在本工作区直接使用已安装 PyTorch 的 rc3_learning/.venv/Scripts/python.exe。
在其他机器，先建立自己的 Python 环境并运行：
python -m pip install -r rc3_improve/requirements.txt
python -B -m rc3_improve.plan --scale 10x10 --mode blocked --budget 8 --output output/scene_plan.json

训练缓存是从场景生成的派生数据，未打包（原工作区仍保留）。如需重新训练，先按 README 生成新的缓存，再训练；训练/测试案例清单和种子已随包保存。
路径有效性只覆盖当前二维导航与代理可达性模型，不代表真实机械臂轨迹已验证。
'''
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('README_START.md',start)
        for path in sorted(files):
            if not path.is_file():raise FileNotFoundError(path)
            archive.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(target) as archive:
        error=archive.testzip()
        if error:raise ValueError(error)
        print(f'{target.name}: {len(archive.namelist())} files; {target.stat().st_size} bytes; CRC OK')

if __name__=='__main__':main()
