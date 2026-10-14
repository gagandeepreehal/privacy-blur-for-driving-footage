"""Small descriptor-relative filesystem primitives shared by safety-critical flows."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import secrets
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC_UNLINKAT: Any | None
try:
    _LIBC_UNLINKAT = _LIBC.unlinkat
    _LIBC_UNLINKAT.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    _LIBC_UNLINKAT.restype = ctypes.c_int
except AttributeError:
    _LIBC_UNLINKAT = None


@dataclass(frozen=True, slots=True)
class StableFileFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int
    sha256: str


class StableFileLockUnavailable(BlockingIOError):
    """A non-blocking stable file lock is already owned."""


def fingerprint_regular_fd(descriptor: int) -> StableFileFingerprint:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise OSError("file is not a single-link regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise OSError("file changed while hashing")
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if before_identity != after_identity:
        raise OSError("file changed while hashing")
    return StableFileFingerprint(*after_identity, digest.hexdigest())


def open_verified_regular(
    parent_fd: int,
    name: str,
    expected: StableFileFingerprint,
    *,
    allow_rename_ctime: bool = False,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        actual = fingerprint_regular_fd(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        exact = (
            actual.device,
            actual.inode,
            actual.mode,
            actual.size,
            actual.mtime_ns,
            actual.ctime_ns,
            actual.nlink,
            actual.sha256,
        )
        expected_exact = (
            expected.device,
            expected.inode,
            expected.mode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.nlink,
            expected.sha256,
        )
        stable_without_ctime = exact[:5] + exact[6:]
        expected_without_ctime = expected_exact[:5] + expected_exact[6:]
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        )
        actual_identity = exact[:7]
        if (
            (exact != expected_exact and not (
                allow_rename_ctime and stable_without_ctime == expected_without_ctime
            ))
            or named_identity != actual_identity
        ):
            raise OSError("file fingerprint or named identity changed")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def unlink_verified_regular(
    parent_fd: int,
    name: str,
    expected: StableFileFingerprint,
) -> None:
    descriptor = open_verified_regular(parent_fd, name, expected)
    try:
        # No Python/project callback is permitted between this final validation and
        # the raw descriptor-relative unlink.
        final = fingerprint_regular_fd(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if final != expected or (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        ) != (
            final.device,
            final.inode,
            final.mode,
            final.size,
            final.mtime_ns,
            final.ctime_ns,
            final.nlink,
        ):
            raise OSError("file changed immediately before unlink")
        raw_unlinkat(parent_fd, strict_basename_bytes(name))
        if os.fstat(descriptor).st_nlink != 0:
            raise OSError("unlinked file retained a foreign hardlink")
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)


def strict_basename_bytes(name: str) -> bytes:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError("filesystem target must be a strict basename")
    return os.fsencode(name)


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        raise OSError(f"required {name} file-opening protection is unavailable")
    return value


def _lock_directory_path(path: Path) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("lock directory path must be canonical and absolute")
    if sys.platform == "darwin" and len(path.parts) > 1:
        system_aliases = {"var", "tmp", "etc"}
        if path.parts[1] in system_aliases:
            return Path("/private", *path.parts[1:])
    return path


def _validate_lock_directory_chain(
    descriptors: list[int], components: tuple[str, ...]
) -> None:
    for parent_fd, component, child_fd in zip(
        descriptors[:-1], components, descriptors[1:], strict=True
    ):
        held = os.fstat(child_fd)
        named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or held.st_nlink < 1
            or (held.st_dev, held.st_ino, held.st_mode)
            != (named.st_dev, named.st_ino, named.st_mode)
        ):
            raise OSError("lock directory identity changed while acquiring it")


def _open_lock_directory_chain(
    path: Path, *, create: bool
) -> tuple[list[int], tuple[str, ...]]:
    physical = _lock_directory_path(path)
    nofollow = _required_open_flag("O_NOFOLLOW")
    directory = _required_open_flag("O_DIRECTORY")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptors = [os.open(physical.anchor, flags)]
    components = physical.parts[1:]
    try:
        for component in components:
            strict_basename_bytes(component)
            try:
                child_fd = os.open(component, flags, dir_fd=descriptors[-1])
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptors[-1])
                    os.fsync(descriptors[-1])
                except FileExistsError:
                    pass
                child_fd = os.open(component, flags, dir_fd=descriptors[-1])
            descriptors.append(child_fd)
            _validate_lock_directory_chain(
                descriptors, components[: len(descriptors) - 1]
            )
        return descriptors, components
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _validate_named_lock(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    held = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or held.st_uid != os.geteuid()
        or stat.S_IMODE(held.st_mode) != 0o400
        or (held.st_dev, held.st_ino) != (expected_device, expected_inode)
        or (held.st_dev, held.st_ino, held.st_mode, held.st_nlink)
        != (named.st_dev, named.st_ino, named.st_mode, named.st_nlink)
    ):
        raise OSError("lock is not a stable single-link regular file")


@contextmanager
def stable_file_lock(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
    expected_parent_device: int,
    expected_parent_inode: int,
    blocking: bool = True,
) -> Iterator[None]:
    """Lock one pre-materialized immutable inode through its physical chain."""

    strict_basename_bytes(path.name)
    directory_fds, components = _open_lock_directory_chain(path.parent, create=False)
    lock_fd: int | None = None
    acquired = False
    try:
        _validate_lock_directory_chain(directory_fds, components)
        parent = os.fstat(directory_fds[-1])
        if (
            (parent.st_dev, parent.st_ino)
            != (expected_parent_device, expected_parent_inode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) != 0o500
        ):
            raise OSError("lock parent identity differs from the lease")
        nofollow = _required_open_flag("O_NOFOLLOW")
        common_flags = (
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        lock_fd = os.open(path.name, common_flags, dir_fd=directory_fds[-1])
        _validate_named_lock(
            directory_fds[-1],
            path.name,
            lock_fd,
            expected_device=expected_device,
            expected_inode=expected_inode,
        )
        _validate_lock_directory_chain(directory_fds, components)
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock_fd, operation)
        except BlockingIOError as error:
            if blocking:
                raise
            raise StableFileLockUnavailable(
                "stable file lock is already owned"
            ) from error
        acquired = True
        try:
            _validate_named_lock(
                directory_fds[-1],
                path.name,
                lock_fd,
                expected_device=expected_device,
                expected_inode=expected_inode,
            )
            _validate_lock_directory_chain(directory_fds, components)
            parent = os.fstat(directory_fds[-1])
            if (
                (parent.st_dev, parent.st_ino)
                != (expected_parent_device, expected_parent_inode)
                or parent.st_uid != os.geteuid()
                or stat.S_IMODE(parent.st_mode) != 0o500
            ):
                raise OSError("lock parent identity differs from the lease")
        except OSError as error:
            raise OSError("lock path changed while acquiring it") from error
        yield
    finally:
        try:
            if acquired and lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            for descriptor in reversed(directory_fds):
                os.close(descriptor)


def raw_unlinkat(directory_fd: int, name: bytes, *, flags: int = 0) -> None:
    if _LIBC_UNLINKAT is None:
        raise OSError("libc unlinkat is unavailable on this platform")
    result = _LIBC_UNLINKAT(directory_fd, name, flags)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), os.fsdecode(name))


def atomic_move_noreplace_between(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    """Atomically move one entry between retained directories without clobbering."""

    source = strict_basename_bytes(source_name)
    destination = strict_basename_bytes(destination_name)
    if sys.platform == "darwin":
        function = _LIBC.renameatx_np
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        flag = 0x00000004
    elif sys.platform.startswith("linux"):
        try:
            function = _LIBC.renameat2
        except AttributeError as error:
            raise OSError("renameat2 RENAME_NOREPLACE is unavailable") from error
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        flag = 0x00000001
    else:
        raise OSError("atomic no-replace move is unsupported on this platform")
    function.restype = ctypes.c_int
    result = function(source_fd, source, destination_fd, destination, flag)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def atomic_move_noreplace(
    directory_fd: int, source_name: str, destination_name: str
) -> None:
    atomic_move_noreplace_between(
        directory_fd, source_name, directory_fd, destination_name
    )


def open_physical_directory(path: Path, *, create: bool = False) -> int:
    """Open a directory by walking every component without following symlinks."""

    absolute = path.absolute()
    if sys.platform == "darwin" and len(absolute.parts) > 1:
        system_aliases = {"var", "tmp", "etc"}
        if absolute.parts[1] in system_aliases:
            absolute = Path("/private", *absolute.parts[1:])
    directory_flag = _required_open_flag("O_DIRECTORY")
    nofollow_flag = _required_open_flag("O_NOFOLLOW")
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | directory_flag | nofollow_flag,
    )
    try:
        for component in absolute.parts[1:]:
            strict_basename_bytes(component)
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | directory_flag | nofollow_flag,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    os.O_RDONLY | directory_flag | nofollow_flag,
                    dir_fd=descriptor,
                )
            held = os.fstat(child)
            named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held.st_mode)
                or (held.st_dev, held.st_ino, held.st_mode)
                != (named.st_dev, named.st_ino, named.st_mode)
            ):
                os.close(child)
                raise OSError("physical directory identity changed while opening")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def open_regular_read(path: Path) -> int:
    """Open one regular file through a retained, symlink-free parent."""

    strict_basename_bytes(path.name)
    parent_fd = open_physical_directory(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | _required_open_flag("O_NOFOLLOW")
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        try:
            held = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(held.st_mode)
                or (held.st_dev, held.st_ino, held.st_mode, held.st_nlink)
                != (named.st_dev, named.st_ino, named.st_mode, named.st_nlink)
            ):
                raise OSError("regular file identity changed while opening")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def read_regular_bytes_at(parent_fd: int, name: str) -> bytes:
    """Read stable bytes from one single-link file below a retained directory."""

    strict_basename_bytes(name)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW")
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        expected = fingerprint_regular_fd(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        expected_identity = (
            expected.device,
            expected.inode,
            expected.mode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.nlink,
        )
        if (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        ) != expected_identity:
            raise OSError("regular file identity changed while reading")
        payload = bytearray()
        offset = 0
        while offset < expected.size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, expected.size - offset), offset
            )
            if not chunk:
                raise OSError("file changed while reading")
            payload.extend(chunk)
            offset += len(chunk)
        if (
            fingerprint_regular_fd(descriptor) != expected
            or len(payload) != expected.size
            or hashlib.sha256(payload).hexdigest() != expected.sha256
        ):
            raise OSError("file changed while reading")
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_mode,
            named_after.st_size,
            named_after.st_mtime_ns,
            named_after.st_ctime_ns,
            named_after.st_nlink,
        ) != expected_identity:
            raise OSError("regular file name changed while reading")
        return bytes(payload)
    finally:
        os.close(descriptor)


def fingerprint_regular_at(parent_fd: int, name: str) -> StableFileFingerprint:
    """Fingerprint one named regular file below a retained directory."""

    strict_basename_bytes(name)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW")
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        fingerprint = fingerprint_regular_fd(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        ) != (
            fingerprint.device,
            fingerprint.inode,
            fingerprint.mode,
            fingerprint.size,
            fingerprint.mtime_ns,
            fingerprint.ctime_ns,
            fingerprint.nlink,
        ):
            raise OSError("regular file identity changed while fingerprinting")
        return fingerprint
    finally:
        os.close(descriptor)


def read_regular_bytes(path: Path) -> bytes:
    """Read stable bytes from one single-link file through its physical path."""

    descriptor = open_regular_read(path)
    try:
        expected = fingerprint_regular_fd(descriptor)
        payload = bytearray()
        offset = 0
        while offset < expected.size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, expected.size - offset), offset
            )
            if not chunk:
                raise OSError("file changed while reading")
            payload.extend(chunk)
            offset += len(chunk)
        if (
            fingerprint_regular_fd(descriptor) != expected
            or len(payload) != expected.size
            or hashlib.sha256(payload).hexdigest() != expected.sha256
        ):
            raise OSError("file changed while reading")
    finally:
        os.close(descriptor)
    current_descriptor = open_regular_read(path)
    try:
        if fingerprint_regular_fd(current_descriptor) != expected:
            raise OSError("file path changed while reading")
    finally:
        os.close(current_descriptor)
    return bytes(payload)


class RetainedRegularFile:
    """A single-link regular file retained by parent and file descriptors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._parent_fd = -1
        self._descriptor = -1
        self._parent_identity: tuple[int, int, int] | None = None
        self._fingerprint: StableFileFingerprint | None = None

    @property
    def fingerprint(self) -> StableFileFingerprint:
        if self._fingerprint is None:
            raise RuntimeError("regular file is not retained")
        return self._fingerprint

    def __enter__(self) -> RetainedRegularFile:
        strict_basename_bytes(self.path.name)
        parent_fd = open_physical_directory(self.path.parent)
        descriptor = -1
        try:
            descriptor = os.open(
                self.path.name,
                os.O_RDONLY
                | _required_open_flag("O_NOFOLLOW")
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            parent = os.fstat(parent_fd)
            fingerprint = fingerprint_regular_fd(descriptor)
            named = os.stat(
                self.path.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                named.st_dev,
                named.st_ino,
                named.st_mode,
                named.st_size,
                named.st_mtime_ns,
                named.st_ctime_ns,
                named.st_nlink,
            ) != (
                fingerprint.device,
                fingerprint.inode,
                fingerprint.mode,
                fingerprint.size,
                fingerprint.mtime_ns,
                fingerprint.ctime_ns,
                fingerprint.nlink,
            ):
                raise OSError("regular file binding changed while retaining it")
            self._parent_fd = parent_fd
            self._descriptor = descriptor
            self._parent_identity = (parent.st_dev, parent.st_ino, parent.st_mode)
            self._fingerprint = fingerprint
            self.revalidate()
            return self
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
            raise

    def duplicate_descriptor(self) -> int:
        self.revalidate()
        descriptor = os.dup(self._descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor

    def revalidate(self) -> None:
        expected = self.fingerprint
        if self._parent_fd < 0 or self._descriptor < 0:
            raise RuntimeError("regular file is not retained")
        held = os.fstat(self._descriptor)
        if (
            held.st_dev,
            held.st_ino,
            held.st_mode,
            held.st_size,
            held.st_mtime_ns,
            held.st_ctime_ns,
            held.st_nlink,
        ) != (
            expected.device,
            expected.inode,
            expected.mode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.nlink,
        ):
            raise OSError("retained regular file changed")
        named = os.stat(
            self.path.name, dir_fd=self._parent_fd, follow_symlinks=False
        )
        if (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        ) != (
            expected.device,
            expected.inode,
            expected.mode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.nlink,
        ):
            raise OSError("retained regular file path binding changed")
        current_parent_fd = open_physical_directory(self.path.parent)
        try:
            current_parent = os.fstat(current_parent_fd)
        finally:
            os.close(current_parent_fd)
        if (
            current_parent.st_dev,
            current_parent.st_ino,
            current_parent.st_mode,
        ) != self._parent_identity:
            raise OSError("retained regular file parent binding changed")

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback
        descriptor, parent_fd = self._descriptor, self._parent_fd
        self._descriptor = -1
        self._parent_fd = -1
        self._fingerprint = None
        self._parent_identity = None
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)


@contextmanager
def atomic_binary_writer(
    path: Path,
    *,
    replace: bool = False,
    expected_prior: StableFileFingerprint | None = None,
) -> Iterator[BinaryIO]:
    """Publish create-only, or replace one explicitly fingerprinted owned file."""

    if replace != (expected_prior is not None):
        raise ValueError("replacement requires an explicit expected prior fingerprint")

    strict_basename_bytes(path.name)
    parent_fd = open_physical_directory(path.parent, create=True)
    temporary_name = f".{path.name}.partial-{secrets.token_hex(16)}"
    strict_basename_bytes(temporary_name)
    descriptor = -1
    stream: BinaryIO | None = None
    committed = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        stream = os.fdopen(descriptor, "wb", closefd=False)
        yield stream
        stream.flush()
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        named = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or (held.st_dev, held.st_ino, held.st_mode, held.st_nlink)
            != (named.st_dev, named.st_ino, named.st_mode, named.st_nlink)
        ):
            raise OSError("atomic output identity changed before publication")
        stream.close()
        stream = None
        readable_temporary = os.open(
            temporary_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            new_fingerprint = fingerprint_regular_fd(readable_temporary)
        finally:
            os.close(readable_temporary)
        if expected_prior is None:
            atomic_move_noreplace(parent_fd, temporary_name, path.name)
        else:
            replaced_descriptor = open_verified_regular(
                parent_fd, path.name, expected_prior
            )
            quarantine_name = f".{path.name}.replaced-{secrets.token_hex(16)}"
            strict_basename_bytes(quarantine_name)
            quarantined = False
            try:
                atomic_move_noreplace(parent_fd, path.name, quarantine_name)
                quarantined = True
                moved = fingerprint_regular_fd(replaced_descriptor)
                named_moved = os.stat(
                    quarantine_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (
                    (moved.device, moved.inode, moved.mode, moved.size, moved.mtime_ns,
                     moved.nlink, moved.sha256)
                    != (expected_prior.device, expected_prior.inode,
                        expected_prior.mode, expected_prior.size,
                        expected_prior.mtime_ns, expected_prior.nlink,
                        expected_prior.sha256)
                    or (named_moved.st_dev, named_moved.st_ino)
                    != (moved.device, moved.inode)
                ):
                    raise OSError("atomic replace target changed during quarantine")
                try:
                    atomic_move_noreplace(parent_fd, temporary_name, path.name)
                except Exception:
                    # Restore only when the canonical name is still absent.  A
                    # foreign insertion wins safely and the owned prior remains
                    # quarantined for explicit reconciliation.
                    try:
                        atomic_move_noreplace(parent_fd, quarantine_name, path.name)
                        quarantined = False
                    except FileExistsError:
                        pass
                    raise
                published_descriptor = open_verified_regular(
                    parent_fd,
                    path.name,
                    new_fingerprint,
                    allow_rename_ctime=True,
                )
                try:
                    published_fingerprint = fingerprint_regular_fd(
                        published_descriptor
                    )
                    named_published = os.stat(
                        path.name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if (
                        published_fingerprint.device,
                        published_fingerprint.inode,
                        published_fingerprint.mode,
                        published_fingerprint.size,
                        published_fingerprint.mtime_ns,
                        published_fingerprint.nlink,
                        published_fingerprint.sha256,
                    ) != (
                        new_fingerprint.device,
                        new_fingerprint.inode,
                        new_fingerprint.mode,
                        new_fingerprint.size,
                        new_fingerprint.mtime_ns,
                        new_fingerprint.nlink,
                        new_fingerprint.sha256,
                    ) or (named_published.st_dev, named_published.st_ino) != (
                        published_fingerprint.device,
                        published_fingerprint.inode,
                    ):
                        raise OSError(
                            "atomic replacement changed before verification"
                        )
                    os.fsync(parent_fd)
                finally:
                    os.close(published_descriptor)
                moved_after = fingerprint_regular_fd(replaced_descriptor)
                unlink_verified_regular(
                    parent_fd, quarantine_name, moved_after
                )
                quarantined = False
            finally:
                os.close(replaced_descriptor)
                if quarantined:
                    os.fsync(parent_fd)
            published = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                published.st_dev,
                published.st_ino,
                published.st_mode,
                published.st_nlink,
            ) != (held.st_dev, held.st_ino, held.st_mode, held.st_nlink):
                raise OSError("atomic output changed during publication")
        committed = True
        os.fsync(parent_fd)
    finally:
        try:
            if stream is not None:
                stream.close()
            if not committed:
                try:
                    raw_unlinkat(parent_fd, strict_basename_bytes(temporary_name))
                    os.fsync(parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)


@contextmanager
def temporary_regular_file(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    before_create: Callable[[str], None] | None = None,
    after_create: Callable[[str, os.stat_result], None] | None = None,
) -> Iterator[tuple[Path, int]]:
    """Create and clean one temporary file through a retained directory."""

    parent_fd = open_physical_directory(directory)
    name = f"{prefix}{secrets.token_hex(16)}{suffix}"
    strict_basename_bytes(name)
    descriptor = -1
    created: os.stat_result | None = None
    try:
        if before_create is not None:
            before_create(name)
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise OSError("temporary file is not a single-link regular file")
        if after_create is not None:
            after_create(name, created)
        yield directory / name, descriptor
    finally:
        try:
            if descriptor >= 0 and created is not None:
                held = os.fstat(descriptor)
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    held.st_nlink != 1
                    or (held.st_dev, held.st_ino, held.st_mode)
                    != (named.st_dev, named.st_ino, named.st_mode)
                    or (held.st_dev, held.st_ino)
                    != (created.st_dev, created.st_ino)
                ):
                    raise OSError("temporary file identity changed before cleanup")
                raw_unlinkat(parent_fd, strict_basename_bytes(name))
                if os.fstat(descriptor).st_nlink != 0:
                    raise OSError("temporary file retained a foreign hardlink")
                os.fsync(parent_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
