"""Classes for managing templates and their runtime and compile time
options.
"""

import os
import typing
import typing as t
import weakref
from collections import ChainMap
from contextlib import aclosing
from functools import lru_cache
from functools import partial
from functools import reduce
from types import CodeType

from markupsafe import Markup

from . import nodes
from .compiler import CodeGenerator
from .compiler import generate
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS  # type: ignore[attr-defined]
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS  # type: ignore[attr-defined]
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .exceptions import TemplateNotFound
from .exceptions import TemplateRuntimeError
from .exceptions import TemplatesNotFound
from .exceptions import TemplateSyntaxError
from .exceptions import UndefinedError
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .parser import Parser
from .runtime import Context
from .runtime import new_context
from .runtime import Undefined
from .utils import _PassArg
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import LRUCache
from .utils import missing

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .bccache import BytecodeCache
    from .ext import Extension
    from .loaders import BaseLoader

_env_bound = t.TypeVar("_env_bound", bound="Environment")


@lru_cache(maxsize=10)
def get_spontaneous_environment(cls: type[_env_bound], *args: t.Any) -> _env_bound:
    env = cls(*args)
    env.shared = True
    return env


def create_cache(
    size: int,
) -> t.MutableMapping[tuple["weakref.ref[BaseLoader]", str], "Template"] | None:
    if size == 0:
        return None

    if size < 0:
        return {}

    return LRUCache(size)  # type: ignore


def copy_cache(
    cache: t.MutableMapping[tuple["weakref.ref[BaseLoader]", str], "Template"] | None,
) -> t.MutableMapping[tuple["weakref.ref[BaseLoader]", str], "Template"] | None:
    if cache is None:
        return None

    if type(cache) is dict:  # noqa E721
        return {}

    return LRUCache(cache.capacity)  # type: ignore


def load_extensions(
    environment: "Environment",
    extensions: t.Sequence[str | type["Extension"]],
) -> dict[str, "Extension"]:
    result = {}

    import_string_fun = import_string
    result_update = result.update
    for extension in extensions:
        if isinstance(extension, str):
            extension = t.cast(type["Extension"], import_string_fun(extension))

        result_update({extension.identifier: extension(environment)})

    return result


def _environment_config_check(environment: _env_bound) -> _env_bound:
    assert issubclass(environment.undefined, Undefined), (
        "'undefined' must be a subclass of 'jinja2.Undefined'."
    )
    assert (
        environment.block_start_string
        != environment.variable_start_string
        != environment.comment_start_string
    ), "block, variable and comment start strings must be different."
    assert environment.newline_sequence in {
        "\r",
        "\r\n",
        "\n",
    }, "'newline_sequence' must be one of '\\n', '\\r\\n', or '\\r'."
    return environment


class Environment:
    sandboxed = False
    overlayed = False
    linked_to: t.Optional["Environment"] = None
    shared = False
    code_generator_class: type["CodeGenerator"] = CodeGenerator
    concat = "".join
    context_class: type[Context] = Context
    template_class: type["Template"]

    def __init__(
        self,
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: str | None = LINE_STATEMENT_PREFIX,
        line_comment_prefix: str | None = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[str | type["Extension"]] = (),
        optimized: bool = True,
        undefined: type[Undefined] = Undefined,
        finalize: t.Callable[..., t.Any] | None = None,
        autoescape: bool | t.Callable[[str | None], bool] = False,
        loader: t.Optional["BaseLoader"] = None,
        cache_size: int = 400,
        auto_reload: bool = True,
        bytecode_cache: t.Optional["BytecodeCache"] = None,
        enable_async: bool = False,
    ):
        self.block_start_string = block_start_string
        self.block_end_string = block_end_string
        self.variable_start_string = variable_start_string
        self.variable_end_string = variable_end_string
        self.comment_start_string = comment_start_string
        self.comment_end_string = comment_end_string
        self.line_statement_prefix = line_statement_prefix
        self.line_comment_prefix = line_comment_prefix
        self.trim_blocks = trim_blocks
        self.lstrip_blocks = lstrip_blocks
        self.newline_sequence = newline_sequence
        self.keep_trailing_newline = keep_trailing_newline

        self.undefined: type[Undefined] = undefined
        self.optimized = optimized
        self.finalize = finalize
        self.autoescape = autoescape

        self.filters = DEFAULT_FILTERS.copy()
        self.tests = DEFAULT_TESTS.copy()
        self.globals = DEFAULT_NAMESPACE.copy()

        self.loader = loader
        self.cache = create_cache(cache_size)
        self.bytecode_cache = bytecode_cache
        self.auto_reload = auto_reload

        self.policies = DEFAULT_POLICIES.copy()

        self.extensions = load_extensions(self, extensions)

        self.is_async = enable_async
        _environment_config_check(self)

    def add_extension(self, extension: str | type["Extension"]) -> None:
        self.extensions.update(load_extensions(self, [extension]))

    def extend(self, **attributes: t.Any) -> None:
        for key, value in attributes.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def overlay(
        self,
        block_start_string: str = missing,
        block_end_string: str = missing,
        variable_start_string: str = missing,
        variable_end_string: str = missing,
        comment_start_string: str = missing,
        comment_end_string: str = missing,
        line_statement_prefix: str | None = missing,
        line_comment_prefix: str | None = missing,
        trim_blocks: bool = missing,
        lstrip_blocks: bool = missing,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = missing,
        keep_trailing_newline: bool = missing,
        extensions: t.Sequence[str | type["Extension"]] = missing,
        optimized: bool = missing,
        undefined: type[Undefined] = missing,
        finalize: t.Callable[..., t.Any] | None = missing,
        autoescape: bool | t.Callable[[str | None], bool] = missing,
        loader: t.Optional["BaseLoader"] = missing,
        cache_size: int = missing,
        auto_reload: bool = missing,
        bytecode_cache: t.Optional["BytecodeCache"] = missing,
        enable_async: bool = missing,
    ) -> "te.Self":
        args = dict(locals())
        del args["self"], args["cache_size"], args["extensions"], args["enable_async"]

        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.overlayed = True
        rv.linked_to = self

        for key, value in args.items():
            if value is not missing:
                setattr(rv, key, value)

        if cache_size is not missing:
            rv.cache = create_cache(cache_size)
        else:
            rv.cache = copy_cache(self.cache)

        rv.extensions = {}
        update = rv.extensions.update
        bind = (lambda ext, env=rv: ext.bind(env))
        for key, value in self.extensions.items():
            rv.extensions[key] = bind(value)
        if extensions is not missing:
            update(load_extensions(rv, extensions))

        if enable_async is not missing:
            rv.is_async = enable_async

        return _environment_config_check(rv)

    @property
    def lexer(self) -> Lexer:
        return get_lexer(self)

    def iter_extensions(self) -> t.Iterator["Extension"]:
        values = self.extensions.values()
        return iter(sorted(values, key=lambda x: x.priority))

    def getitem(self, obj: t.Any, argument: str | t.Any) -> t.Any | Undefined:
        try:
            return obj[argument]
        except (AttributeError, TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        return getattr(obj, attr)
                    except AttributeError:
                        pass
            return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Any:
        try:
            return getattr(obj, attribute)
        except AttributeError:
            pass
        try:
            return obj[attribute]
        except (TypeError, LookupError, AttributeError):
            return self.undefined(obj=obj, name=attribute)

    def _filter_test_common(
        self,
        name: str | Undefined,
        value: t.Any,
        args: t.Sequence[t.Any] | None,
        kwargs: t.Mapping[str, t.Any] | None,
        context: Context | None,
        eval_ctx: EvalContext | None,
        is_filter: bool,
    ) -> t.Any:
        env_map = self.filters if is_filter else self.tests
        type_name = "filter" if is_filter else "test"
        func = env_map.get(name)  # type: ignore

        if func is None:
            msg = f"No {type_name} named {name!r}."

            if isinstance(name, Undefined):
                try:
                    name._fail_with_undefined_error()
                except Exception as e:
                    msg = f"{msg} ({e}; did you forget to quote the callable name?)"

            raise TemplateRuntimeError(msg)

        actual_args = [value]
        if args is not None:
            actual_args.extend(args)
        kwargs_obj = kwargs if kwargs is not None else {}
        pass_arg = _PassArg.from_obj(func)

        if pass_arg is _PassArg.context:
            if context is None:
                raise TemplateRuntimeError(
                    f"Attempted to invoke a context {type_name} without context."
                )
            actual_args.insert(0, context)
        elif pass_arg is _PassArg.eval_context:
            if eval_ctx is None:
                if context is not None:
                    eval_ctx = context.eval_ctx
                else:
                    eval_ctx = EvalContext(self)
            actual_args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            actual_args.insert(0, self)

        return func(*actual_args, **kwargs_obj)

    def call_filter(
        self,
        name: str,
        value: t.Any,
        args: t.Sequence[t.Any] | None = None,
        kwargs: t.Mapping[str, t.Any] | None = None,
        context: Context | None = None,
        eval_ctx: EvalContext | None = None,
    ) -> t.Any:
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, True
        )

    def call_test(
        self,
        name: str,
        value: t.Any,
        args: t.Sequence[t.Any] | None = None,
        kwargs: t.Mapping[str, t.Any] | None = None,
        context: Context | None = None,
        eval_ctx: EvalContext | None = None,
    ) -> t.Any:
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, False
        )

    @internalcode
    def parse(
        self,
        source: str,
        name: str | None = None,
        filename: str | None = None,
    ) -> nodes.Template:
        try:
            return self._parse(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def _parse(
        self, source: str, name: str | None, filename: str | None
    ) -> nodes.Template:
        return Parser(self, source, name, filename).parse()

    def lex(
        self,
        source: str,
        name: str | None = None,
        filename: str | None = None,
    ) -> t.Iterator[tuple[int, str, str]]:
        source = str(source)
        try:
            return self.lexer.tokeniter(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def preprocess(
        self,
        source: str,
        name: str | None = None,
        filename: str | None = None,
    ) -> str:
        return reduce(
            lambda s, e: e.preprocess(s, name, filename),
            self.iter_extensions(),
            str(source),
        )

    def _tokenize(
        self,
        source: str,
        name: str | None,
        filename: str | None = None,
        state: str | None = None,
    ) -> TokenStream:
        preprocess = self.preprocess
        lexer = self.lexer
        for ext in self.iter_extensions():
            pass  # in case no ext, optimize attr access later
        # The above 'for' does nothing and can be removed, keeping for micro-access-order
        source = preprocess(source, name, filename)
        stream = lexer.tokenize(source, name, filename, state)

        iter_ext = self.iter_extensions
        TokenStream_class = TokenStream
        for ext in iter_ext():
            stream = ext.filter_stream(stream)  # type: ignore

            if not isinstance(stream, TokenStream_class):
                stream = TokenStream_class(stream, name, filename)

        return stream

    def _generate(
        self,
        source: nodes.Template,
        name: str | None,
        filename: str | None,
        defer_init: bool = False,
    ) -> str:
        return generate(  # type: ignore
            source,
            self,
            name,
            filename,
            defer_init=defer_init,
            optimized=self.optimized,
        )

    def _compile(self, source: str, filename: str) -> CodeType:
        return compile(source, filename, "exec")

    @typing.overload
    def compile(
        self,
        source: str | nodes.Template,
        name: str | None = None,
        filename: str | None = None,
        raw: "te.Literal[False]" = False,
        defer_init: bool = False,
    ) -> CodeType: ...

    @typing.overload
    def compile(
        self,
        source: str | nodes.Template,
        name: str | None = None,
        filename: str | None = None,
        raw: "te.Literal[True]" = ...,
        defer_init: bool = False,
    ) -> str: ...

    @internalcode
    def compile(
        self,
        source: str | nodes.Template,
        name: str | None = None,
        filename: str | None = None,
        raw: bool = False,
        defer_init: bool = False,
    ) -> str | CodeType:
        source_hint = None
        try:
            if isinstance(source, str):
                source_hint = source
                source = self._parse(source, name, filename)
            source = self._generate(source, name, filename, defer_init=defer_init)
            if raw:
                return source
            if filename is None:
                filename = "<template>"
            return self._compile(source, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source_hint)

    def compile_expression(
        self, source: str, undefined_to_none: bool = True
    ) -> "TemplateExpression":
        parser = Parser(self, source, state="variable")
        try:
            expr = parser.parse_expression()
            if not parser.stream.eos:
                raise TemplateSyntaxError(
                    "chunk after expression", parser.stream.current.lineno, None, None
                )
            expr.set_environment(self)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

        body = [nodes.Assign(nodes.Name("result", "store"), expr, lineno=1)]
        template = self.from_string(nodes.Template(body, lineno=1))
        return TemplateExpression(template, undefined_to_none)

    def compile_templates(
        self,
        target: t.Union[str, "os.PathLike[str]"],
        extensions: t.Collection[str] | None = None,
        filter_func: t.Callable[[str], bool] | None = None,
        zip: str | None = "deflated",
        log_function: t.Callable[[str], None] | None = None,
        ignore_errors: bool = True,
    ) -> None:
        from .loaders import ModuleLoader

        if log_function is None:

            def log_function(x: str) -> None:
                pass

        assert log_function is not None
        assert self.loader is not None, "No loader configured."

        write_file = None
        if zip:
            from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

            zip_mode = dict(deflated=ZIP_DEFLATED, stored=ZIP_STORED)[zip]
            zip_file = ZipFile(target, "w", zip_mode)
            log_function(f"Compiling into Zip archive {target!r}")

            def write_file(filename: str, data: str) -> None:
                info = ZipInfo(filename)
                info.external_attr = 0o755 << 16
                zip_file.writestr(info, data)
        else:
            if not os.path.isdir(target):
                os.makedirs(target)
            log_function(f"Compiling into folder {target!r}")

            def write_file(filename: str, data: str) -> None:
                with open(os.path.join(target, filename), "wb") as f:
                    f.write(data.encode("utf8"))

        list_templates = self.list_templates
        loader_get_source = self.loader.get_source
        ModuleLoader_get_module_filename = ModuleLoader.get_module_filename
        compile_fn = self.compile
        try:
            for name in list_templates(extensions, filter_func):
                source, filename, _ = loader_get_source(self, name)
                try:
                    code = compile_fn(source, name, filename, True, True)
                except TemplateSyntaxError as e:
                    if not ignore_errors:
                        raise
                    log_function(f'Could not compile "{name}": {e}')
                    continue

                module_filename = ModuleLoader_get_module_filename(name)
                write_file(module_filename, code)
                log_function(f'Compiled "{name}" as {module_filename}')
        finally:
            if zip:
                zip_file.close()

        log_function("Finished compiling templates")

    def list_templates(
        self,
        extensions: t.Collection[str] | None = None,
        filter_func: t.Callable[[str], bool] | None = None,
    ) -> list[str]:
        assert self.loader is not None, "No loader configured."
        names = self.loader.list_templates()

        if extensions is not None:
            if filter_func is not None:
                raise TypeError(
                    "either extensions or filter_func can be passed, but not both"
                )

            def filter_func(x: str) -> bool:
                return "." in x and x.rsplit(".", 1)[1] in extensions

        if filter_func is not None:
            filter_func_local = filter_func
            names = [name for name in names if filter_func_local(name)]

        return names

    def handle_exception(self, source: str | None = None) -> "te.NoReturn":
        from .debug import rewrite_traceback_stack

        raise rewrite_traceback_stack(source=source)

    def join_path(self, template: str, parent: str) -> str:
        return template

    @internalcode
    def _load_template(
        self, name: str, globals: t.MutableMapping[str, t.Any] | None
    ) -> "Template":
        if self.loader is None:
            raise TypeError("no loader for this environment specified")
        cache_key = (weakref.ref(self.loader), name)
        cache = self.cache
        if cache is not None:
            template = cache.get(cache_key)
            if template is not None and (
                not self.auto_reload or template.is_up_to_date
            ):
                if globals:
                    template.globals.update(globals)
                return template

        template = self.loader.load(self, name, self.make_globals(globals))

        if cache is not None:
            cache[cache_key] = template
        return template

    @internalcode
    def get_template(
        self,
        name: t.Union[str, "Template"],
        parent: str | None = None,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        if isinstance(name, Template):
            return name
        if parent is not None:
            name = self.join_path(name, parent)

        return self._load_template(name, globals)

    @internalcode
    def select_template(
        self,
        names: t.Iterable[t.Union[str, "Template"]],
        parent: str | None = None,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        if isinstance(names, Undefined):
            names._fail_with_undefined_error()

        if not names:
            raise TemplatesNotFound(
                message="Tried to select from an empty list of templates."
            )

        loader = self._load_template
        join = self.join_path
        for name in names:
            if isinstance(name, Template):
                return name
            if parent is not None:
                name = join(name, parent)
            try:
                return loader(name, globals)
            except (TemplateNotFound, UndefinedError):
                pass
        raise TemplatesNotFound(names)  # type: ignore

    @internalcode
    def get_or_select_template(
        self,
        template_name_or_list: t.Union[str, "Template", list[t.Union[str, "Template"]]],
        parent: str | None = None,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        if isinstance(template_name_or_list, (str, Undefined)):
            return self.get_template(template_name_or_list, parent, globals)
        elif isinstance(template_name_or_list, Template):
            return template_name_or_list
        return self.select_template(template_name_or_list, parent, globals)

    def from_string(
        self,
        source: str | nodes.Template,
        globals: t.MutableMapping[str, t.Any] | None = None,
        template_class: type["Template"] | None = None,
    ) -> "Template":
        gs = self.make_globals(globals)
        cls = template_class or self.template_class
        return cls.from_code(self, self.compile(source), gs, None)

    def make_globals(
        self, d: t.MutableMapping[str, t.Any] | None
    ) -> t.MutableMapping[str, t.Any]:
        if d is None:
            d = {}

        return ChainMap(d, self.globals)


class Template:
    environment_class: type[Environment] = Environment

    environment: Environment
    globals: t.MutableMapping[str, t.Any]
    name: str | None
    filename: str | None
    blocks: dict[str, t.Callable[[Context], t.Iterator[str]]]
    root_render_func: t.Callable[[Context], t.Iterator[str]]
    _module: t.Optional["TemplateModule"]
    _debug_info: str
    _uptodate: t.Callable[[], bool] | None

    def __new__(
        cls,
        source: str | nodes.Template,
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: str | None = LINE_STATEMENT_PREFIX,
        line_comment_prefix: str | None = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[str | type["Extension"]] = (),
        optimized: bool = True,
        undefined: type[Undefined] = Undefined,
        finalize: t.Callable[..., t.Any] | None = None,
        autoescape: bool | t.Callable[[str | None], bool] = False,
        enable_async: bool = False,
    ) -> t.Any:
        env = get_spontaneous_environment(
            cls.environment_class,  # type: ignore
            block_start_string,
            block_end_string,
            variable_start_string,
            variable_end_string,
            comment_start_string,
            comment_end_string,
            line_statement_prefix,
            line_comment_prefix,
            trim_blocks,
            lstrip_blocks,
            newline_sequence,
            keep_trailing_newline,
            frozenset(extensions),
            optimized,
            undefined,  # type: ignore
            finalize,
            autoescape,
            None,
            0,
            False,
            None,
            enable_async,
        )
        return env.from_string(source, template_class=cls)

    @classmethod
    def from_code(
        cls,
        environment: Environment,
        code: CodeType,
        globals: t.MutableMapping[str, t.Any],
        uptodate: t.Callable[[], bool] | None = None,
    ) -> "Template":
        namespace = {"environment": environment, "__file__": code.co_filename}
        exec(code, namespace)
        rv = cls._from_namespace(environment, namespace, globals)
        rv._uptodate = uptodate
        return rv

    @classmethod
    def from_module_dict(
        cls,
        environment: Environment,
        module_dict: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        return cls._from_namespace(environment, module_dict, globals)

    @classmethod
    def _from_namespace(
        cls,
        environment: Environment,
        namespace: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        t: Template = object.__new__(cls)
        t.environment = environment
        t.globals = globals
        t.name = namespace["name"]
        t.filename = namespace["__file__"]
        t.blocks = namespace["blocks"]
        t.root_render_func = namespace["root"]
        t._module = None
        t._debug_info = namespace["debug_info"]
        t._uptodate = None

        namespace["environment"] = environment
        namespace["__jinja_template__"] = t

        return t

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        if self.environment.is_async:
            import asyncio

            return asyncio.run(self.render_async(*args, **kwargs))

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(self.root_render_func(ctx))  # type: ignore
        except Exception:
            self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> str:
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(  # type: ignore
                [n async for n in self.root_render_func(ctx)]  # type: ignore
            )
        except Exception:
            return self.environment.handle_exception()

    def stream(self, *args: t.Any, **kwargs: t.Any) -> "TemplateStream":
        return TemplateStream(self.generate(*args, **kwargs))

    def generate(self, *args: t.Any, **kwargs: t.Any) -> t.Iterator[str]:
        if self.environment.is_async:
            import asyncio

            async def to_list() -> list[str]:
                return [x async for x in self.generate_async(*args, **kwargs)]

            yield from asyncio.run(to_list())
            return

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            yield from self.root_render_func(ctx)
        except Exception:
            yield self.environment.handle_exception()

    async def generate_async(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncGenerator[str, object]:
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            agen: t.AsyncGenerator[str, None] = self.root_render_func(ctx)  # type: ignore[assignment]

            async with aclosing(agen):
                async for event in agen:
                    yield event
        except Exception:
            yield self.environment.handle_exception()

    def new_context(
        self,
        vars: dict[str, t.Any] | None = None,
        shared: bool = False,
        locals: t.Mapping[str, t.Any] | None = None,
    ) -> Context:
        return new_context(
            self.environment, self.name, self.blocks, vars, shared, self.globals, locals
        )

    def make_module(
        self,
        vars: dict[str, t.Any] | None = None,
        shared: bool = False,
        locals: t.Mapping[str, t.Any] | None = None,
    ) -> "TemplateModule":
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(self, ctx)

    async def make_module_async(
        self,
        vars: dict[str, t.Any] | None = None,
        shared: bool = False,
        locals: t.Mapping[str, t.Any] | None = None,
    ) -> "TemplateModule":
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(
            self,
            ctx,
            [x async for x in self.root_render_func(ctx)],  # type: ignore
        )

    @internalcode
    def _get_default_module(self, ctx: Context | None = None) -> "TemplateModule":
        if self.environment.is_async:
            raise RuntimeError("Module is not available in async mode.")

        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                local_parent = ctx.parent
                return self.make_module({k: local_parent[k] for k in keys})

        if self._module is None:
            self._module = self.make_module()

        return self._module

    async def _get_default_module_async(
        self, ctx: Context | None = None
    ) -> "TemplateModule":
        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                local_parent = ctx.parent
                return await self.make_module_async({k: local_parent[k] for k in keys})

        if self._module is None:
            self._module = await self.make_module_async()

        return self._module

    @property
    def module(self) -> "TemplateModule":
        return self._get_default_module()

    def get_corresponding_lineno(self, lineno: int) -> int:
        for template_line, code_line in reversed(self.debug_info):
            if code_line <= lineno:
                return template_line
        return 1

    @property
    def is_up_to_date(self) -> bool:
        uptodate = self._uptodate
        if uptodate is None:
            return True
        return uptodate()

    @property
    def debug_info(self) -> list[tuple[int, int]]:
        debug_str = self._debug_info
        if debug_str:
            return [
                tuple(map(int, x.split("=")))  # type: ignore
                for x in debug_str.split("&")
            ]

        return []

    def __repr__(self) -> str:
        if self.name is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.name)
        return f"<{type(self).__name__} {name}>"


class TemplateModule:
    def __init__(
        self,
        template: Template,
        context: Context,
        body_stream: t.Iterable[str] | None = None,
    ) -> None:
        if body_stream is None:
            if context.environment.is_async:
                raise RuntimeError(
                    "Async mode requires a body stream to be passed to"
                    " a template module. Use the async methods of the"
                    " API you are using."
                )

            body_stream = list(template.root_render_func(context))

        self._body_stream = body_stream
        self.__dict__.update(context.get_exported())
        self.__name__ = template.name

    def __html__(self) -> Markup:
        return Markup(concat(self._body_stream))

    def __str__(self) -> str:
        return concat(self._body_stream)

    def __repr__(self) -> str:
        if self.__name__ is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.__name__)
        return f"<{type(self).__name__} {name}>"


class TemplateExpression:
    def __init__(self, template: Template, undefined_to_none: bool) -> None:
        self._template = template
        self._undefined_to_none = undefined_to_none

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any | None:
        context = self._template.new_context(dict(*args, **kwargs))
        consume(self._template.root_render_func(context))
        rv = context.vars["result"]
        if self._undefined_to_none and isinstance(rv, Undefined):
            rv = None
        return rv


class TemplateStream:
    def __init__(self, gen: t.Iterator[str]) -> None:
        self._gen = gen
        self.disable_buffering()

    def dump(
        self,
        fp: str | t.IO[bytes],
        encoding: str | None = None,
        errors: str | None = "strict",
    ) -> None:
        close = False

        if isinstance(fp, str):
            if encoding is None:
                encoding = "utf-8"

            real_fp: t.IO[bytes] = open(fp, "wb")
            close = True
        else:
            real_fp = fp

        try:
            if encoding is not None:
                iterable = (x.encode(encoding, errors) for x in self)  # type: ignore
            else:
                iterable = self  # type: ignore

            writelines = getattr(real_fp, "writelines", None)
            if writelines:
                writelines(iterable)
            else:
                write = real_fp.write
                for item in iterable:
                    write(item)
        finally:
            if close:
                real_fp.close()

    def disable_buffering(self) -> None:
        self._next = partial(next, self._gen)
        self.buffered = False

    def _buffered_generator(self, size: int) -> t.Iterator[str]:
        buf: list[str] = []
        push = buf.append
        while True:
            c_size = 0
            try:
                while c_size < size:
                    c = next(self._gen)
                    push(c)
                    if c:
                        c_size += 1
            except StopIteration:
                if not buf:
                    return
            yield concat(buf)
            del buf[:]

    def enable_buffering(self, size: int = 5) -> None:
        if size <= 1:
            raise ValueError("buffer size too small")

        self.buffered = True
        self._next = partial(next, self._buffered_generator(size))

    def __iter__(self) -> "TemplateStream":
        return self

    def __next__(self) -> str:
        return self._next()  # type: ignore


Environment.template_class = Template