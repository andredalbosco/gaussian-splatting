import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/ellipsoid-benchmark')
groups = defaultdict(list)
for row in csv.DictReader((root / 'results.csv').open()):
    groups[row['scene'], int(row['width']), row['method']].append(row)

with (root / 'summary.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['scene', 'width', 'method', 'gpu_us', 'gpu_trial_min_us', 'gpu_trial_max_us',
                     'batch_us', 'max_color_changed', 'max_coverage_changed', 'max_depth_changed'])
    for (scene, width, method), rows in groups.items():
        times = [float(row['gpu_median_ms']) * 1000 for row in rows]
        writer.writerow([scene, width, method, statistics.median(times), min(times), max(times),
                         statistics.median(float(row['batch_ms']) * 1000 for row in rows),
                         *(max(int(row[key]) for row in rows) for key in
                           ['color_changed', 'coverage_changed', 'depth_changed'])])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['box_vertex_general', 'box_culled_vertex_general', 'box_quad_vertex_general',
           'exact_quad_vertex_general', 'golus_vertex_general']
labels = ['Box, both sides', 'Box, front culled', 'Projected box quad', 'Exact ellipsoid quad', 'Golus billboard']
colors = ['#aeb9c5', '#cf9551', '#9478af', '#3686c4', '#52957b']
selected = sys.argv[2:] or ['luigi', 'luigi_close']
fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), layout='constrained', squeeze=False)
for ax, scene in zip(axes[0], selected):
    title = scene.replace('_', ' ')
    values = []
    errors = [[], []]
    for method in methods:
        times = [float(row['gpu_median_ms']) * 1000 for row in groups[scene, 2560, method]]
        median = statistics.median(times)
        values.append(median)
        errors[0].append(median - min(times))
        errors[1].append(max(times) - median)
    ax.barh(labels, values, xerr=errors, color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('GPU render pass (µs), lower is faster')
    ax.spines[['top', 'right']].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value + max(values) * .025, index, f'{value:.1f}', va='center', fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
fig.suptitle('RTX 4090 · Dawn/Vulkan · 2560×1440\nVertex rays / general intersection; bars include clear and store', fontsize=12)
fig.savefig(root / 'comparison.png', dpi=180)

from PIL import Image, ImageDraw, ImageFont
scenes = selected if sys.argv[2:] else ['close', 'grid_ellipsoids', 'luigi_close']
methods = ['box_culled_vertex_general', 'box_quad_vertex_general', 'exact_quad_vertex_general',
           'golus_vertex_general', 'ellipse', 'gaussian']
labels = ['Box (culled)', 'Box quad', 'Exact quad', 'Golus', 'Projected ellipse', 'Gaussian']
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
if not (root / f'{scenes[0]}_box_culled_vertex_general.ppm').exists():
    print(f'{len(groups)} summary rows written; comparison.png saved (source PPMs unavailable)')
    sys.exit(0)
sheet = Image.new('RGB', (6 * 320, len(scenes) * 220), '#141923')
draw = ImageDraw.Draw(sheet)
for y, scene in enumerate(scenes):
    for x, (method, label) in enumerate(zip(methods, labels)):
        image = Image.open(root / f'{scene}_{method}.ppm').resize((320, 180))
        sheet.paste(image, (x * 320, y * 220 + 35))
        draw.text((x * 320 + 6, y * 220 + 8), f'{scene}: {label}', font=font, fill='#e0e6ef')
sheet.save(root / 'contact-sheet.png')
print(f'{len(groups)} summary rows written; comparison.png and contact-sheet.png saved')
