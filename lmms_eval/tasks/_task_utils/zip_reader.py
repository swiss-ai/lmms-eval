import os
import threading
import zipfile
from typing import Callable, Optional, Union

PathLike = Union[str, os.PathLike[str]]


class ThreadLocalZipReader:
    """Read ZIP members through one ZipFile handle per worker thread."""

    def __init__(self, path_fn: Callable[[], PathLike]):
        self._path_fn = path_fn
        self._path: Optional[str] = None
        self._path_lock = threading.Lock()
        self._local = threading.local()

    def _get_path(self) -> str:
        path = self._path
        if path is None:
            with self._path_lock:
                if self._path is None:
                    self._path = os.fspath(self._path_fn())
                path = self._path
        return path

    def _get_archive(self) -> zipfile.ZipFile:
        archive = getattr(self._local, "archive", None)
        archive_path = getattr(self._local, "archive_path", None)
        path = self._get_path()
        if archive is None or archive_path != path:
            if archive is not None:
                archive.close()
            archive = zipfile.ZipFile(path, "r")
            self._local.archive = archive
            self._local.archive_path = path
        return archive

    def read(self, member: str) -> bytes:
        with self._get_archive().open(member) as fp:
            return fp.read()
