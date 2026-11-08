"""Read a supported sequential text ZMX and evaluate local glass catalogs.

This input adapter loads fixed, coaxial, transmissive prescriptions in mm.
It leaves field mapping and ray launch to the standalone evaluator.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


FIELD_TYPES = {0: "angle", 1: "object_height", 3: "real_image_height"}
UNSUPPORTED_APERTURES = {"SQAP", "ELAP", "OBSC", "UDAP", "UDAD", "USAP", "APDX", "APDY", "OBDC", "DECA"}


def _decode(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _finite(value):
    number = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(number):
        raise ValueError("Expected a finite prescription value")
    return number


@dataclass(frozen=True)
class _Glass:
    formula: int
    coefficients: tuple[float, ...]
    bounds_um: tuple[float, ...]
    reference_temperature_c: float

    def index(self, wavelength_um):
        if self.bounds_um and not self.bounds_um[0] <= wavelength_um <= self.bounds_um[1]:
            raise ValueError(f"Wavelength {wavelength_um:g} µm outside catalog validity range")
        if self.formula not in (1, 2) or len(self.coefficients) < 6:
            raise ValueError(f"Unsupported AGF dispersion formula {self.formula}")
        a, b, c, d, e, f = self.coefficients[:6]
        wavelength_squared = wavelength_um * wavelength_um
        if self.formula == 1:
            inverse = 1.0 / wavelength_squared
            squared = a + b * wavelength_squared + inverse * (c + inverse * (d + inverse * (e + inverse * f)))
        else:
            terms = [1.0]
            for strength, resonance in ((a,b), (c,d), (e,f)):
                if strength == 0.0:
                    continue
                denominator = wavelength_squared - resonance
                if denominator == 0.0:
                    raise ValueError("Wavelength is at a Sellmeier resonance")
                terms.append(strength * wavelength_squared / denominator)
            squared = math.fsum(terms)
        if not math.isfinite(squared) or squared <= 0.0:
            raise ValueError("AGF formula returned an invalid refractive index")
        return math.sqrt(squared)


def _read_catalog(path, names):
    glasses = {}
    current = None
    records = {}

    def finish():
        if current is None:
            return
        bounds = tuple(map(_finite, records.get("LD", ())[:2]))
        if bounds and (len(bounds) != 2 or not 0 < bounds[0] < bounds[1]):
            raise ValueError(f"Invalid wavelength bounds for {current[0]} in {path.name}")
        thermal = records.get("TD", ())
        glasses[current[0]] = _Glass(current[1], tuple(map(_finite, records.get("CD", ()))),
                                      bounds, _finite(thermal[6]) if len(thermal) >= 7 else 20.0)

    for line in _decode(path.read_bytes()).splitlines():
        columns = line.split()
        if not columns:
            continue
        if columns[0] == "NM":
            finish()
            current = (columns[1].upper(), int(columns[2])) if len(columns) > 2 and columns[1].upper() in names else None
            records = {}
        elif current is not None and columns[0] in ("CD", "LD", "TD"):
            records[columns[0]] = columns[1:]
    finish()
    return glasses


def _resolve_glasses(materials, declared, glass_directory):
    required = {name for name, _ in materials if name is not None}
    if not required:
        return {}, {}
    directory = Path(glass_directory)
    catalogs = {p.stem.upper():p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() == ".agf"}
    records = {}

    def find(catalog, name):
        if catalog not in catalogs:
            return None
        if catalog not in records:
            records[catalog] = _read_catalog(catalogs[catalog], required)
        return records[catalog].get(name)

    glasses, selected = {}, {}
    for name, explicit in dict.fromkeys(materials):
        if name is None:
            continue
        preferred = tuple(dict.fromkeys(([explicit] if explicit else []) + list(declared)))
        for catalog in preferred:
            glass = find(catalog, name)
            if glass is not None:
                break
        else:
            matches = [(catalog, value) for catalog in catalogs if (value := find(catalog, name)) is not None]
            if not matches:
                raise ValueError(f"Catalog material {name} was not found")
            catalog, glass = matches[0]
            if any(value != glass for _, value in matches):
                raise ValueError(f"Catalog material {name} is ambiguous outside the declared catalogs")
        glasses[(name,explicit)] = glass
        selected[(name,explicit)] = catalog
    return glasses, selected


def _surface_row(number, records):
    surface_type = records.get("TYPE", [""])[0]
    if surface_type not in ("STANDARD", "EVENASPH"):
        raise ValueError(f"Unsupported surface type {surface_type!r} at surface {number}")
    if UNSUPPORTED_APERTURES.intersection(records):
        raise ValueError(f"Unsupported aperture at surface {number}")
    if any(tag in records for tag in ("PARX", "TILT", "DCOX", "DCOY", "NSS", "GRIN")):
        raise ValueError(f"Unsupported noncoaxial or gradient geometry at surface {number}")
    mirror = records.get("MIRR")
    if mirror is not None and mirror != ["2", "0"]:
        raise ValueError(f"Unsupported mirror record at surface {number}")
    material = records.get("GLAS", [])
    name = material[0].upper() if material else None
    if name in {"AIR", "VACUUM"}:
        name = None
    if name in {"MIRROR", "REFLECT", "___BLANK"} or (name and len(material) > 1 and material[1] != "0"):
        raise ValueError(f"Unsupported reflection or noncatalog material at surface {number}")
    explicit_catalog = material[11].upper() if name and len(material) > 11 else None
    thickness_text = records.get("DISZ", ["0"])[0]
    thickness = math.inf if number == 0 and thickness_text.upper() == "INFINITY" else _finite(thickness_text)
    if thickness < 0.0:
        raise ValueError(f"Unsupported negative thickness at surface {number}")
    aperture_radius = None
    if "CLAP" in records:
        aperture = list(map(_finite, records["CLAP"]))
        if len(aperture) < 2 or aperture[0] != 0.0 or aperture[1] <= 0.0 or any(aperture[2:]):
            raise ValueError(f"Unsupported annular or decentered circular aperture at surface {number}")
        aperture_radius = aperture[1]
    parameters = records.get("PARM", {})
    if surface_type == "STANDARD" and any(parameters.values()):
        raise ValueError(f"Unsupported standard-surface parameters at surface {number}")
    coefficients = tuple(parameters.get(i,0.0) for i in range(1,max(parameters,default=0)+1)) if surface_type == "EVENASPH" else ()
    row = {"curvature_mm_inv":_finite(records.get("CURV",["0"])[0]),
           "conic":_finite(records.get("CONI",["0"])[0]),
           "even_coefficients":coefficients, "aperture_radius_mm":aperture_radius}
    # Saved display/mechanical radii describe the solid geometry. CLAP alone
    # continues to control the existing optical ray-clipping convention.
    for record, key in (("DIAM", "clear_radius"), ("MEMA", "mechanical_radius")):
        values = records.get(record)
        if values is None:
            continue
        if not values:
            raise ValueError(f"Missing {record} radius at surface {number}")
        radius = _finite(values[0])
        if radius < 0.:
            raise ValueError(f"Negative {record} radius at surface {number}")
        if radius:
            row[key+"_mm"] = radius
            mode = values[1] if len(values) > 1 else None
            row[key+"_source"] = "zmx_saved_automatic" if mode == "0" else "zmx_saved_declared"
            row[key+"_mode"] = mode
    return row, (name,explicit_catalog), thickness, surface_type


def load_zmx(path, glass_directory):
    """Load fixed coaxial text ZMX geometry and evaluate active wavelengths.

    Vertex and surface arrays start at surface 1. Index arrays include surface 0.
    Wavelength slots in the return value, including primary_wave, are zero-based.
    An infinite object puts surface 1 at z=0; a finite object puts surface 0 at z=0.
    """
    raw = Path(path).read_bytes()
    header, waves, surfaces = {}, {}, {}
    current = None
    for line in _decode(raw).splitlines():
        columns = line.split()
        if not columns:
            continue
        tag, *values = columns
        if tag == "SURF" and not line[0].isspace():
            current = int(values[0])
            if current in surfaces:
                raise ValueError(f"Duplicate surface {current}")
            surfaces[current] = {}
        elif current is not None and line[0].isspace():
            if tag == "PARM":
                order = int(values[0])
                if not 1 <= order <= 32:
                    raise ValueError(f"Unsupported even-asphere term at surface {current}")
                surfaces[current].setdefault(tag,{})[order] = _finite(values[1])
            else:
                surfaces[current][tag] = values
        elif not surfaces:
            if tag == "WAVM":
                waves[int(values[0])] = (_finite(values[1]), _finite(values[2]))
            else:
                header[tag] = values
        else:
            current = None
            if tag == "MCON" or (tag == "CONF" and int(values[0]) > 1):
                raise ValueError("Multiple configurations require an explicit configuration adapter")

    if header.get("MODE") != ["SEQ"] or header.get("UNIT",[""])[0] != "MM":
        raise ValueError("Use a sequential text ZMX in millimeters")
    if int(header.get("MNUM",["1"])[0]) != 1:
        raise ValueError("Multiple configurations require an explicit configuration adapter")
    if any(_finite(value) != 0.0 for value in header.get("RAIM", ["0","0"])[:2]):
        raise ValueError("Active Zemax ray aiming requires an explicit launch adapter")
    apodization = list(map(_finite,header.get("GFAC",["0","0"])[:2]))
    if len(apodization)!=2 or apodization[0] not in (0.,1.) or apodization[1]<0:
        raise ValueError("Only uniform and Gaussian pupil apodization are supported")
    if header.get("ROPD",["2"])[0] != "2":
        raise ValueError("Use the exit-pupil OPD reference")
    if len(surfaces) < 3 or sorted(surfaces) != list(range(len(surfaces))):
        raise ValueError("Surface numbers must be contiguous from object through image")
    ftyp = list(map(int,header.get("FTYP",[])))
    if (len(ftyp) < 5 or ftyp[0] not in FIELD_TYPES or ftyp[1] not in (0,1)
            or ftyp[4] not in (0,1) or min(ftyp[2:4]) < 1):
        raise ValueError("Unsupported field or wavelength definition")
    field_count, wave_count = ftyp[2:4]
    field_x,field_y = (list(map(_finite,header.get(tag,[])[:field_count])) for tag in ("XFLN","YFLN"))
    if len(field_x) != field_count or len(field_y) != field_count:
        raise ValueError("Incomplete field records")
    fields = list(zip(field_x,field_y))
    vignetting = [list(map(_finite,header.get(tag,[])[:field_count])) for tag in ("VDXN","VDYN","VCXN","VCYN","VANN")]
    for column in vignetting:
        column.extend([0.0]*(field_count-len(column)))
    if any(any(column) for column in vignetting):
        raise ValueError("Field vignetting factors require an explicit pupil mapping adapter")
    if any(slot not in waves for slot in range(1,wave_count+1)):
        raise ValueError("Incomplete active wavelength slots")
    wavelengths, weights = map(list, zip(*(waves[i] for i in range(1,wave_count+1))))
    primary = int(header.get("PWAV",["1"])[0])-1
    if not 0 <= primary < wave_count or min(wavelengths) <= 0 or min(weights) < 0 or max(weights) <= 0:
        raise ValueError("Invalid primary wavelength or wavelength values")
    aperture_keys = [key for key in ("OBNA","ENPD") if key in header]
    if len(aperture_keys) != 1:
        raise ValueError("Only Object Space NA and Entrance Pupil Diameter are supported")
    aperture_type = aperture_keys[0].lower()
    aperture_value = _finite(header[aperture_keys[0]][0])
    if aperture_value <= 0.0:
        raise ValueError("Aperture value must be positive")
    stop = [number for number, records in surfaces.items() if "STOP" in records]
    if len(stop) != 1 or not 0 < stop[0] < len(surfaces)-1:
        raise ValueError("Expected one stop between object and image")
    parsed = [_surface_row(i,surfaces[i]) for i in range(len(surfaces))]
    rows,materials,thicknesses,types = map(list,zip(*parsed))
    if types[0] != "STANDARD" or rows[0]["curvature_mm_inv"] != 0.0:
        raise ValueError("An explicit object surface shape requires an object adapter")
    at_infinity = math.isinf(thicknesses[0])
    if at_infinity and aperture_type != "enpd":
        raise ValueError("An infinite object requires an entrance-pupil diameter")
    position = 0.0 if at_infinity else thicknesses[0]
    vertices = [position]
    for thickness in thicknesses[1:-1]:
        position += thickness
        vertices.append(position)

    declared = [name.upper() for name in header.get("GCAT",[])]
    glasses,selected = _resolve_glasses(materials,declared,glass_directory)
    environment = list(map(_finite,header.get("ENVD",["20","1","0"])[:3]))
    if len(environment) != 3:
        raise ValueError("Incomplete environment record")
    temperature, pressure, adjust_indices = environment
    if adjust_indices and (temperature != 20.0 or pressure != 1.0 or any(g.reference_temperature_c != temperature for g in glasses.values())):
        raise ValueError("Thermal or pressure-dependent indices require an environment adapter")
    indices = [[1.0 if material[0] is None else glasses[material].index(wavelength)
                for material in materials] for wavelength in wavelengths]
    if aperture_type == "obna" and aperture_value >= indices[primary][0]:
        raise ValueError("Object-space NA must be below the object medium index")
    return {"source_sha256":hashlib.sha256(raw).hexdigest(), "vertices_mm":vertices,
            "surface_rows":rows[1:], "surface_types":types[1:], "indices_by_wave":indices,
            "wavelengths_um":wavelengths, "wavelength_weights":weights, "primary_wave":primary,
            "stop_surface":stop[0], "object_at_infinity":at_infinity,
            "field_type":FIELD_TYPES[ftyp[0]], "fields":fields,
            "field_normalization":"radial" if ftyp[4] == 0 else "rectangular",
            "object_space_telecentric":bool(ftyp[1]),
            "gaussian_apodization":apodization[1] if apodization[0]==1. else 0.,
            "field_vignetting":list(zip(*vignetting)), "aperture_type":aperture_type,
            "aperture_value":aperture_value, "material_names":[m[0] for m in materials],
            "material_catalogs":[selected.get(m) for m in materials],
            "catalogs_resolved_outside_header":sorted(set(selected.values())-set(declared))}
