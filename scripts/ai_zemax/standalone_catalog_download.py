"""Explicit official-catalog download, hash verification and local indexing.

Raw vendor data stay in a caller-selected private directory. Registry hashes
pin the reviewed release; a changed server response is not silently accepted.
Network access is only used by an explicit call/CLI, never during ray tracing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile

from .standalone_glass_catalog import import_catalogs

SCHEMA = 'standalone_optics.official_catalog_sources.v1'
REGISTRY = Path(__file__).resolve().parents[2]/'references/glass-catalog-sources.json'
HOSTS = {'www.cdgmgd.com', 'cdgmgd.com', 'media.schott.com', 'www.schott.com',
         'www.ohara-inc.co.jp', 'ohara-inc.co.jp'}
MAX_BYTES = 16*1024*1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _url(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.username
            or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Catalog download requires a registered official HTTPS host')
    return url


class _OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, target):
        _url(target)
        return super().redirect_request(request, response, code, message, headers, target)


def _download(url):
    request = Request(_url(url), headers={'User-Agent': 'standalone-optics/catalog-import'})
    with build_opener(_OfficialRedirect()).open(request, timeout=60) as response:
        _url(response.geturl())
        raw = response.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Catalog download exceeds the size limit')
    return raw


def _filename(name):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or PurePosixPath(name).name != name or PureWindowsPath(name).name != name
            or PureWindowsPath(name).drive or name.rstrip(' .') != name
            or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name)
            or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$',
                *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}):
        raise ValueError('Catalog filenames must be safe basenames')
    return name


def _private_path(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Local cache path escapes the selected private directory')
    return path


def _write_exact(path, raw, expected):
    if _sha(raw) != expected:
        raise ValueError('Official catalog hash differs from the pinned reviewed release')
    if path.exists():
        if _sha(path.read_bytes()) != expected:
            raise ValueError('Existing local catalog file was changed: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def _zip_members(raw):
    archive = ZipFile(BytesIO(raw))
    total = 0
    names = set()
    for item in archive.infolist():
        posix, windows = PurePosixPath(item.filename), PureWindowsPath(item.filename)
        mode = item.external_attr >> 16
        if (posix.is_absolute() or windows.is_absolute() or windows.drive
                or '..' in posix.parts or '..' in windows.parts or mode & 0o170000 == 0o120000):
            archive.close()
            raise ValueError('Unsafe path or symbolic link in catalog archive')
        identity = item.filename.replace('\\', '/').casefold()
        if identity in names:
            archive.close()
            raise ValueError('Duplicate member name in catalog archive')
        names.add(identity)
        total += item.file_size
        if item.file_size > MAX_BYTES or total > 4*MAX_BYTES:
            archive.close()
            raise ValueError('Expanded catalog archive exceeds the size limit')
    return archive


def fetch_catalogs(destination, *, registry_path=REGISTRY, include_historical=False,
                   source_ids=None, cached_downloads=None, index_name='glass-index.json', transport=None):
    """Import pinned official releases, from HTTPS or an identified local cache.

The returned index references relative content-addressed snapshots. It can be
copied with its ``catalog-snapshots`` directory to another computer. Existing
indexes are never replaced. ``transport`` is a test hook returning bytes.
    """
    destination = Path(destination).expanduser().resolve()
    index = _private_path(destination, _filename(index_name))
    receipt_path = _private_path(destination, index.name+'.download-receipt.json')
    _private_path(destination, 'catalog-snapshots')
    if index.exists() or receipt_path.exists():
        raise FileExistsError('Select a new destination/index for catalog updates')
    registry = json.loads(Path(registry_path).read_text(encoding='utf-8'))
    if registry.get('schema') != SCHEMA:
        raise ValueError('Unsupported official catalog registry')
    sources = registry['sources']
    if source_ids:
        requested = set(source_ids)
        if requested-{s['source_id'] for s in sources}:
            raise ValueError('Unknown official catalog source identity')
        sources = [s for s in sources if s['source_id'] in requested]
    elif not include_historical:
        sources = [s for s in sources if s['default_current']]
    if not sources:
        raise ValueError('No catalog sources selected')
    downloaded, metadata, inputs = [], {}, []
    for source in sources:
        identity = _filename(source['source_id'])
        name = _filename(source['archive_name'])
        _url(source['download_url'])
        if cached_downloads is not None:
            raw = (Path(cached_downloads)/identity/name).read_bytes()
        else:
            raw = (transport or _download)(source['download_url'])
        if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
            raise ValueError('Download must be bounded catalog bytes')
        folder = _private_path(destination, 'downloads', identity)
        _write_exact(_private_path(destination, 'downloads', identity, name), raw, source['archive_sha256'])
        archive = _zip_members(raw) if name.lower().endswith('.zip') else None
        try:
            for agf in source['agf_files']:
                filename = _filename(agf['filename'])
                if not filename.lower().endswith('.agf'):
                    raise ValueError('Only AGF catalog data can be imported')
                member = agf['archive_member']
                content = archive.read(member) if archive is not None else raw
                if archive is None and member is not None:
                    raise ValueError('A direct AGF cannot declare an archive member')
                path = _private_path(destination, 'downloads', identity, filename)
                _write_exact(path, content, agf['sha256'])
                inputs.append(path)
                metadata[str(path.resolve())] = dict(vendor=source['vendor'], catalog_name=identity,
                    scope=source['scope'], default_current=source['default_current'],
                    version_label=source['version_label'], release_date=source['release_date'],
                    release_date_basis=source['release_date_basis'], official_source_url=source['official_landing_url'],
                    verified_sha256=agf['sha256'], verified_at_utc=source['verified_at_utc'],
                    downloaded_at_utc=(source['verified_at_utc'] if cached_downloads is not None
                                       else datetime.now(timezone.utc).isoformat()),
                    latest_verified=source.get('latest_verified', False),
                    latest_verified_as_of=source.get('latest_verified_as_of', registry['as_of_date']))
                downloaded.append(dict(source_id=identity, path=str(path), sha256=agf['sha256']))
        finally:
            if archive is not None:
                archive.close()
    catalog = import_catalogs(inputs, index, metadata=metadata)
    receipt = dict(registry_as_of=registry['as_of_date'], registry_sha256=_sha(Path(registry_path).read_bytes()),
                   mode='verified_local_cache' if cached_downloads is not None else 'official_https',
                   index=str(index), files=downloaded, raw_data_publicly_bundled=False,
                   imported_at_utc=datetime.now(timezone.utc).isoformat(),
                   latest_source_verification_as_of=registry['as_of_date'])
    with receipt_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    return catalog, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--list-sources', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--include-historical', action='store_true')
    parser.add_argument('--source', action='append')
    parser.add_argument('--from-cache', type=Path)
    parser.add_argument('--index-name', default='glass-index.json')
    args = parser.parse_args(argv)
    if args.list_sources:
        print(args.registry.read_text(encoding='utf-8'))
        return 0
    if args.out is None:
        parser.error('--out is required for an explicit local import')
    catalog, receipt = fetch_catalogs(args.out, registry_path=args.registry,
        include_historical=args.include_historical, source_ids=args.source,
        cached_downloads=args.from_cache, index_name=args.index_name)
    print(json.dumps(dict(index=str(catalog.path), catalog_count=len(catalog.catalogs),
                         sources=[r['source_id'] for r in receipt['files']]), indent=2))
    return 0
