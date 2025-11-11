"""API and implementations for loading templates from different data
sources.
"""

import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType

from .exceptions import TemplateNotFound
from .utils import internalcode

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template


def split_template_path(template: str) -> list[str]:
    pieces = []
    split = template.split("/")
    os_sep = os.sep
    os_altsep = os.path.altsep
    os_pardir = os.path.pardir
    for piece in split:
        if (
            os_sep in piece
            or (os_altsep and os_altsep in piece)
            or piece == os_pardir
        ):
            raise TemplateNotFound(template)
        elif piece and piece != ".":
            pieces.append(piece)
    return pieces


class BaseLoader:
    has_source_access = True

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        if not self.has_source_access:
            raise RuntimeError(
                f"{type(self).__name__} cannot provide access to the source"
            )
        raise TemplateNotFound(template)

    def list_templates(self) -> list[str]:
        raise TypeError("this loader cannot iterate over all templates")

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        code = None
        if globals is None:
            globals = {}

        source, filename, uptodate = self.get_source(environment, name)

        bcc = environment.bytecode_cache
        bucket = None
        if bcc is not None:
            bucket = bcc.get_bucket(environment, name, filename, source)
            code = bucket.code

        if code is None:
            code = environment.compile(source, name, filename)

        if bcc is not None and bucket.code is None:
            bucket.code = code
            bcc.set_bucket(bucket)

        return environment.template_class.from_code(
            environment, code, globals, uptodate
        )


class FileSystemLoader(BaseLoader):
    def __init__(
        self,
        searchpath: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
        encoding: str = "utf-8",
        followlinks: bool = False,
    ) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]

        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str, t.Callable[[], bool]]:
        pieces = split_template_path(template)
        spaths = self.searchpath
        for searchpath in spaths:
            filename = posixpath.join(searchpath, *pieces)
            if os.path.isfile(filename):
                break
        else:
            spaths_len = len(spaths)
            plural = "path" if spaths_len == 1 else "paths"
            paths_str = ", ".join(repr(p) for p in spaths)
            raise TemplateNotFound(
                template,
                f"{template!r} not found in search {plural}: {paths_str}",
            )

        encoding = self.encoding
        with open(filename, encoding=encoding) as f:
            contents = f.read()

        mtime = os.path.getmtime(filename)

        def uptodate() -> bool:
            try:
                return os.path.getmtime(filename) == mtime
            except OSError:
                return False

        return contents, os.path.normpath(filename), uptodate

    def list_templates(self) -> list[str]:
        found = set()
        spaths = self.searchpath
        os_walk = os.walk
        os_path_join = os.path.join
        os_sep = os.sep
        os_strip = os_sep
        os_replace = os_sep
        for searchpath in spaths:
            walk_dir = os_walk(searchpath, followlinks=self.followlinks)
            for dirpath, _, filenames in walk_dir:
                dprefix = dirpath[len(searchpath):]
                for filename in filenames:
                    template = (
                        os_path_join(dirpath, filename)[len(searchpath):]
                        .strip(os_strip)
                        .replace(os_replace, "/")
                    )
                    if template[:2] == "./":
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)


if sys.version_info >= (3, 13):

    def _get_zipimporter_files(z: t.Any) -> dict[str, object]:
        try:
            get_files = z._get_files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required metadata to list templates."
            ) from e
        return get_files()

else:

    def _get_zipimporter_files(z: t.Any) -> dict[str, object]:
        try:
            files = z._files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required metadata to list templates."
            ) from e
        return files  # type: ignore[no-any-return]


class PackageLoader(BaseLoader):
    def __init__(
        self,
        package_name: str,
        package_path: "str" = "templates",
        encoding: str = "utf-8",
    ) -> None:
        os_path_normpath = os.path.normpath
        os_sep = os.sep
        os_curdir = os.path.curdir

        package_path = os_path_normpath(package_path).rstrip(os_sep)

        if package_path == os_curdir:
            package_path = ""
        elif package_path[:2] == os_curdir + os_sep:
            package_path = package_path[2:]

        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding

        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, "An import spec was not found for the package."
        loader = spec.loader
        assert loader is not None, "A loader was not found for the package."
        self._loader = loader
        self._archive = None

        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))  # type: ignore
            template_root = os.path.join(pkgdir, package_path).rstrip(os_sep)
        else:
            roots: list[str] = []
            submodule_search_locations = spec.submodule_search_locations
            if submodule_search_locations:
                roots.extend(submodule_search_locations)
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))

            if not roots:
                raise ValueError(
                    f"The {package_name!r} package was not installed in a"
                    " way that PackageLoader understands."
                )

            template_root = None
            for root in roots:
                root_joined = os.path.join(root, package_path)
                if os.path.isdir(root_joined):
                    template_root = root_joined
                    break
            if template_root is None:
                raise ValueError(
                    f"PackageLoader could not find a {package_path!r} directory"
                    f" in the {package_name!r} package."
                )

        self._template_root = template_root

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str, t.Callable[[], bool] | None]:
        os_path_normpath = os.path.normpath
        posixpath_join = posixpath.join
        path = os_path_normpath(
            posixpath_join(self._template_root, *split_template_path(template))
        )
        p = path
        up_to_date: t.Callable[[], bool] | None

        if self._archive is None:
            if not os.path.isfile(p):
                raise TemplateNotFound(template)

            with open(p, "rb") as f:
                source = f.read()

            mtime = os.path.getmtime(p)

            def up_to_date() -> bool:
                return os.path.isfile(p) and os.path.getmtime(p) == mtime

        else:
            get_data = self._loader.get_data
            try:
                source = get_data(p)  # type: ignore
            except OSError as e:
                raise TemplateNotFound(template) from e

            up_to_date = None

        return source.decode(self.encoding), p, up_to_date

    def list_templates(self) -> list[str]:
        results: list[str] = []
        template_root = self._template_root
        os_sep = os.sep

        if self._archive is None:
            offset = len(template_root)
            os_walk = os.walk
            os_path_join = os.path.join
            os_replace = os_sep
            for dirpath, _, filenames in os_walk(template_root):
                dirpath_sub = dirpath[offset:].lstrip(os_sep)
                for name in filenames:
                    results.append(
                        os_path_join(dirpath_sub, name).replace(os_replace, "/")
                    )
        else:
            files = _get_zipimporter_files(self._loader)
            archive = self._archive
            prefix = template_root[len(archive):].lstrip(os_sep) + os_sep
            offset = len(prefix)
            os_replace = os_sep
            for name in files:
                if name.startswith(prefix) and name[-1] != os_sep:
                    results.append(name[offset:].replace(os_replace, "/"))

        results.sort()
        return results


class DictLoader(BaseLoader):
    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, None, t.Callable[[], bool]]:
        mapping = self.mapping
        if template in mapping:
            source = mapping[template]
            return source, None, lambda: source == mapping.get(template)
        raise TemplateNotFound(template)

    def list_templates(self) -> list[str]:
        return sorted(self.mapping)


class FunctionLoader(BaseLoader):
    def __init__(
        self,
        load_func: t.Callable[
            [str],
            str | tuple[str, str | None, t.Callable[[], bool] | None] | None,
        ],
    ) -> None:
        self.load_func = load_func

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        rv = self.load_func(template)

        if rv is None:
            raise TemplateNotFound(template)

        if isinstance(rv, str):
            return rv, None, None

        return rv


class PrefixLoader(BaseLoader):
    def __init__(
        self, mapping: t.Mapping[str, BaseLoader], delimiter: str = "/"
    ) -> None:
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template: str) -> tuple[BaseLoader, str]:
        mapping = self.mapping
        delimiter = self.delimiter
        try:
            prefix, name = template.split(delimiter, 1)
            loader = mapping[prefix]
        except (ValueError, KeyError) as e:
            raise TemplateNotFound(template) from e
        return loader, name

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound as e:
            raise TemplateNotFound(template) from e

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound as e:
            raise TemplateNotFound(name) from e

    def list_templates(self) -> list[str]:
        result = []
        mapping_items = self.mapping.items()
        delimiter = self.delimiter
        for prefix, loader in mapping_items:
            templates = loader.list_templates()
            for template in templates:
                result.append(prefix + delimiter + template)
        return result


class ChoiceLoader(BaseLoader):
    def __init__(self, loaders: t.Sequence[BaseLoader]) -> None:
        self.loaders = loaders

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        loaders = self.loaders
        for loader in loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        loaders = self.loaders
        for loader in loaders:
            try:
                return loader.load(environment, name, globals)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(name)

    def list_templates(self) -> list[str]:
        found = set()
        loaders = self.loaders
        for loader in loaders:
            found.update(loader.list_templates())
        return sorted(found)


class _TemplateModule(ModuleType):
    pass


class ModuleLoader(BaseLoader):
    has_source_access = False

    def __init__(
        self,
        path: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
    ) -> None:
        package_name = f"_jinja2_module_templates_{id(self):x}"

        mod = _TemplateModule(package_name)

        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]

        mod.__path__ = [os.fspath(p) for p in path]

        sys.modules[package_name] = weakref.proxy(
            mod, lambda x: sys.modules.pop(package_name, None)
        )

        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name: str) -> str:
        return "tmpl_" + sha1(name.encode("utf-8")).hexdigest()

    @staticmethod
    def get_module_filename(name: str) -> str:
        return ModuleLoader.get_template_key(name) + ".py"

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        key = self.get_template_key(name)
        module = f"{self.package_name}.{key}"
        mod_obj = getattr(self.module, module, None)

        if mod_obj is None:
            try:
                mod_obj = __import__(module, None, None, ["root"])
            except ImportError as e:
                raise TemplateNotFound(name) from e

            sys.modules.pop(module, None)

        if globals is None:
            globals = {}

        return environment.template_class.from_module_dict(
            environment, mod_obj.__dict__, globals
        )