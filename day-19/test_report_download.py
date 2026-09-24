import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import report_download as download


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_download_and_overwrite_protection(self):
        content = '# Отчёт\n'.encode()
        saved = {'path':str(download.REMOTE_DIR / 'ssh-2026-09-24_19-00-00.md'),
                 'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch.object(download,'REPORT_DIR',Path(directory)), patch.object(download,'read_remote',AsyncMock(return_value=content)) as read:
            result = await download.download_report(saved)
            path = Path(result['local_path'])
            self.assertEqual(path.read_bytes(),content)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            with self.assertRaises(FileExistsError):
                await download.download_report(saved)
            self.assertEqual(path.read_bytes(),content)
            read.reset_mock()
            with self.assertRaises(ValueError):
                await download.download_report({**saved,'path':'/etc/passwd'})
            read.assert_not_awaited()

    async def test_bad_hash_leaves_no_local_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(download,'REPORT_DIR',Path(directory)), patch.object(download,'read_remote',AsyncMock(return_value=b'bad')):
            with self.assertRaises(ValueError):
                await download.download_report({'path':str(download.REMOTE_DIR / 'report.md'),'bytes':3,'sha256':'0'*64})
            self.assertEqual(list(Path(directory).iterdir()),[])
