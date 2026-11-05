"""Dependency-free SVG views of fixed-image spot and ray-fan results."""

from html import escape
from math import ceil, floor, isfinite, log10
from pathlib import Path


COLORS = ('#17669a', '#b94b29', '#258053', '#8e4da1', '#946b07', '#ba4974', '#246e72')


def _limit(values):
    maximum = max((abs(value) for value in values if value is not None and isfinite(value)), default=0.)
    if maximum == 0.:
        return 1.
    magnitude = 10.**floor(log10(maximum))
    return ceil(maximum*1.02/magnitude*2.)*magnitude/2.


def _text(x, y, value, style='', anchor='start'):
    return (f'<text x="{x:.2f}" y="{y:.2f}" class="{style}" '
            f'text-anchor="{anchor}">{escape(str(value))}</text>')


def _legend(waves, x, y, columns):
    elements = []
    for i, wave in enumerate(waves):
        cx, cy = x+(i % columns)*126, y+(i//columns)*21
        color = COLORS[i % len(COLORS)]
        label = f'{wave["wavelength_um"]*1000.:.6g} nm'
        if wave.get('status') == 'failed':
            label += ' (failed)'
        elements.append(f'<path d="M{cx} {cy-4}h15" stroke="{color}" stroke-width="2"/>')
        elements.append(_text(cx+22, cy, label, 'small'))
    return elements


def _frame(box, limits, labels):
    left, top, width, height = box
    max_x, max_y = limits
    elements = [f'<rect x="{left}" y="{top}" width="{width}" height="{height}" class="frame"/>']
    for fraction in (-1., -.5, 0., .5, 1.):
        x = left+(1.+fraction)*width/2.
        y = top+(1.-fraction)*height/2.
        css = 'origin' if fraction == 0. else 'grid'
        elements.append(f'<path d="M{x:.2f} {top}v{height} M{left} {y:.2f}h{width}" class="{css}"/>')
        elements.append(_text(x, top+height+17, f'{max_x*fraction:.4g}', 'small', 'middle'))
        elements.append(_text(left-9, y+4, f'{max_y*fraction:.4g}', 'small', 'end'))
    elements.append(_text(left+width/2., top+height+37, labels[0], 'axis', 'middle'))
    elements.append(f'<g transform="translate({left-67} {top+height/2.}) rotate(-90)">'
                    +_text(0, 0, labels[1], 'axis', 'middle')+'</g>')
    return elements


def _project(point, box, limits):
    x, y = point
    left, top, width, height = box
    max_x, max_y = limits
    return left+(1.+x/max_x)*width/2., top+(1.-y/max_y)*height/2.


def _save(path, width, height, title, note, elements):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(note)}</desc>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#263448}'
        '.title{font-size:22px;font-weight:600}.sub{font-size:13px;fill:#46566b}'
        '.axis{font-size:12px}.small{font-size:11px}.frame{fill:#f8fafc;stroke:#97a6b5}'
        '.grid{fill:none;stroke:#e0e6ed}.origin{fill:none;stroke:#8396a8;stroke-dasharray:4 3}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(25, 32, title, 'title'),
        _text(25, 54, note, 'sub'),
        *elements,
        '</svg>',
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(svg)+'\n', encoding='utf-8')


def write_spot_svg(path, spot, *, field_index):
    """Write an equal-scale spot diagram for one field."""
    waves = spot['waves']
    maximum = _limit(value for wave in waves for point in wave.get('intercepts_xy_um', ())
                     for value in point)
    box = (95, 88, 435, 435)
    limits = (maximum, maximum)
    elements = _frame(box, limits, ('Image X (µm)', 'Image Y (µm)'))
    for i, wave in enumerate(waves):
        color = COLORS[i % len(COLORS)]
        path_data = []
        for point in wave.get('intercepts_xy_um', ()):
            if all(isfinite(value) for value in point):
                x, y = _project(point, box, limits)
                path_data.append(f'M{x:.2f} {y:.2f}h.01')
        if path_data:
            data = ' '.join(path_data)
            elements.append(f'<path d="{data}" fill="none" stroke="{color}" '
                            'stroke-opacity=".53" stroke-width="2.3" stroke-linecap="round"/>')
    if not any(wave.get('intercepts_xy_um') for wave in waves):
        elements.append(_text(312, 306, 'No valid spot data', 'sub', 'middle'))
    elements.extend(_legend(waves, 95, 590, 4))
    height = 610+21*max(0, (len(waves)-1)//4)
    status = spot['status'].replace('_', ' ')
    note = f'Fixed image plane · primary chief reference · equal axes · {status}'
    _save(path, 640, height, f'Spot diagram · field {field_index}', note, elements)


def _fan_path(coordinates, samples, box, limits):
    path = []
    start = True
    for pupil, sample in zip(coordinates, samples, strict=True):
        if sample is None or not isfinite(sample):
            start = True
            continue
        x, y = _project((pupil, sample), box, limits)
        path.append(f'{"M" if start else "L"}{x:.2f} {y:.2f}')
        start = False
    return ' '.join(path)


def write_ray_fan_svg(path, aberrations, *, field_index):
    """Write fixed-axis T/S ray fans for a one-based saved field index."""
    if not 1 <= field_index <= len(aberrations['fields']):
        raise ValueError('Ray fan field index is outside the analysis')
    field = aberrations['fields'][field_index-1]
    waves = field['waves']
    coordinates = aberrations['fan_pupil_coordinate']
    key = 'primary_chief_referenced_xy_fans_image_um'
    maximum = _limit(sample for wave in waves for fan in wave[key].values() for sample in fan)
    limits = (1., maximum)
    elements = []
    panels = (
        ((95, 111, 320, 260), 'y', 'Tangential fan', 'Normalized pupil Py', 'Image Y error (µm)'),
        ((550, 111, 320, 260), 'x', 'Sagittal fan', 'Normalized pupil Px', 'Image X error (µm)'),
    )
    for box, axis, heading, x_label, y_label in panels:
        left, top, width, height = box
        elements.extend(_frame(box, limits, (x_label, y_label)))
        elements.append(_text(left+width/2., top-13, heading, 'axis', 'middle'))
        for i, wave in enumerate(waves):
            color = COLORS[i % len(COLORS)]
            data = _fan_path(coordinates, wave[key][axis], box, limits)
            if data:
                elements.append(f'<path d="{data}" stroke="{color}" stroke-width="1.7" fill="none"/>')
    elements.extend(_legend(waves, 95, 438, 6))
    height = 458+21*max(0, (len(waves)-1)//6)
    note = 'Fixed image plane · primary chief reference · gaps indicate failed rays'
    if any(wave.get('negative_segment_rays') for wave in waves):
        note += ' · signed comparison only'
    _save(path, 920, height, f'Ray fan · field {field_index}', note, elements)
