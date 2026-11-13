"""Pinned release identity, archive safety and immutable local imports."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.ai_zemax.standalone_catalog_download import fetch_catalogs, _url, _filename


def agf():
    def n(w):
        return math.sqrt(2.25+.004/w**2)
    nd = n(.5875618)
    vd = (nd-1.)/(n(.4861327)-n(.6562725))
    return (f'CC Synthetic test data, not vendor glass\nNM DEMO 1 0 {nd:.12g} {vd:.12g} 0 0 0\n'
            'CD 2.25 0 .004 0 0 0\nLD .3 2.5\n').encode('utf-8')


def registry(path, data, *, zipped=False):
    glass = agf()
    row = dict(source_id='cdgm-current', vendor='CDGM', default_current=True, scope='fixture',
        version_label='TEST', release_date=None, release_date_basis='test fixture',
        official_landing_url='https://www.cdgmgd.com/',
        download_url='https://www.cdgmgd.com/test-data',
        archive_name='catalog.zip' if zipped else 'test.agf',
        archive_sha256=hashlib.sha256(data).hexdigest(), verified_at_utc='2026-11-09T00:00:00Z',
        agf_files=[dict(filename='test.agf', sha256=hashlib.sha256(glass).hexdigest(),
                        archive_member='test.agf' if zipped else None)])
    path.write_text(json.dumps(dict(schema='standalone_optics.official_catalog_sources.v1',
                                   as_of_date='2026-11-09', sources=[row])), encoding='utf-8')
    return row


class CatalogDownloadTests(unittest.TestCase):
    def test_direct_download_is_pinned_and_existing_index_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root/'sources.json'
            registry(meta, agf())
            catalog, receipt = fetch_catalogs(root/'private', registry_path=meta, transport=lambda _: agf())
            self.assertTrue(catalog.path.is_file())
            self.assertEqual(receipt['mode'], 'official_https')
            self.assertFalse(receipt['raw_data_publicly_bundled'])
            self.assertAlmostEqual(catalog.refractive_index('DEMO', .55, vendor='CDGM'),
                                   math.sqrt(2.25+.004/.55**2), places=12)
            with self.assertRaises(FileExistsError):
                fetch_catalogs(root/'private', registry_path=meta,
                               transport=lambda _: self.fail('Must not contact server for an existing index'))

    def test_changed_server_file_never_becomes_a_verified_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry(root/'sources.json', agf())
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                fetch_catalogs(root/'private', registry_path=root/'sources.json', transport=lambda _: b'error page')
            self.assertFalse((root/'private/glass-index.json').exists())

    def test_zip_traversal_is_rejected_and_valid_zip_imports_offline(self):
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                buffer = BytesIO()
                with ZipFile(buffer, 'w') as archive:
                    archive.writestr('test.agf', agf())
                    if unsafe:
                        archive.writestr('../outside.txt', 'unexpected')
                raw = buffer.getvalue()
                registry(root/'sources.json', raw, zipped=True)
                cached = root/'cache/cdgm-current'
                cached.mkdir(parents=True)
                (cached/'catalog.zip').write_bytes(raw)
                if unsafe:
                    with self.assertRaisesRegex(ValueError, 'Unsafe path'):
                        fetch_catalogs(root/'private', registry_path=root/'sources.json', cached_downloads=root/'cache')
                else:
                    catalog, receipt = fetch_catalogs(root/'private', registry_path=root/'sources.json',
                                                      cached_downloads=root/'cache')
                    self.assertEqual(receipt['mode'], 'verified_local_cache')
                    self.assertEqual(len(catalog.catalogs), 1)
                self.assertFalse((root/'outside.txt').exists())

    def test_download_hosts_are_explicit_and_https(self):
        for url in ('http://www.cdgmgd.com/file', 'https://evil.example/file',
                    'https://www.cdgmgd.com.evil.example/file', 'https://user@www.schott.com/file'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _url(url)

    def test_windows_names_cannot_target_devices_or_alternate_streams(self):
        for name in ('test.agf:stream', 'NUL.agf', 'CON', 'COM1.txt', 'file.agf.', 'file.agf '):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _filename(name)

    def test_cached_copy_preserves_original_download_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry(root/'sources.json', agf())
            cache = root/'cache/cdgm-current'
            cache.mkdir(parents=True)
            (cache/'test.agf').write_bytes(agf())
            catalog, _ = fetch_catalogs(root/'private', registry_path=root/'sources.json', cached_downloads=root/'cache')
            release = catalog.catalogs[0]['release']
            self.assertEqual(release['downloaded_at_utc'], '2026-11-09T00:00:00Z')
            self.assertFalse(release['latest_verified'])
            self.assertTrue((root/'private/glass-index.json.download-receipt.json').exists())

    def test_new_index_keeps_previous_receipt_and_zip_names_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry(root/'sources.json', agf())
            fetch_catalogs(root/'private', registry_path=root/'sources.json', transport=lambda _: agf())
            receipt = root/'private/glass-index.json.download-receipt.json'
            before = receipt.read_bytes()
            fetch_catalogs(root/'private', registry_path=root/'sources.json',
                           index_name='second-index.json', transport=lambda _: agf())
            self.assertEqual(receipt.read_bytes(), before)
            self.assertTrue((root/'private/second-index.json.download-receipt.json').exists())
            buffer = BytesIO()
            with ZipFile(buffer, 'w') as archive:
                archive.writestr('test.agf', agf())
                archive.writestr('TEST.AGF', agf())
            raw = buffer.getvalue()
            registry(root/'duplicates.json', raw, zipped=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate member'):
                fetch_catalogs(root/'duplicate-run', registry_path=root/'duplicates.json', transport=lambda _: raw)


if __name__ == '__main__':
    unittest.main()
