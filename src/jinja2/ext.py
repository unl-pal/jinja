"""Extension API for adding custom tags and behavior."""

import pprint
import re
import typing as t

from markupsafe import Markup

from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .lexer import Token
    from .lexer import TokenStream
    from .parser import Parser

    class _TranslationsBasic(te.Protocol):
        def gettext(self, message: str) -> str: ...

        def ngettext(self, singular: str, plural: str, n: int) -> str:
            pass

    class _TranslationsContext(_TranslationsBasic):
        def pgettext(self, context: str, message: str) -> str: ...

        def npgettext(
            self, context: str, singular: str, plural: str, n: int
        ) -> str: ...

    _SupportedTranslations = _TranslationsBasic | _TranslationsContext


GETTEXT_FUNCTIONS: tuple[str, ...] = (
    "_",
    "gettext",
    "ngettext",
    "pgettext",
    "npgettext",
)
_ws_re = re.compile(r"\s*\n\s*")


class Extension:
    identifier: t.ClassVar[str]

    def __init_subclass__(cls) -> None:
        cls.identifier = f"{cls.__module__}.{cls.__name__}"

    tags: set[str] = set()
    priority = 100

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def bind(self, environment: Environment) -> "te.Self":
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.environment = environment
        return rv

    def preprocess(
        self, source: str, name: str | None, filename: str | None = None
    ) -> str:
        return source

    def filter_stream(
        self, stream: "TokenStream"
    ) -> t.Union["TokenStream", t.Iterable["Token"]]:
        return stream

    def parse(self, parser: "Parser") -> nodes.Node | list[nodes.Node]:
        raise NotImplementedError()

    def attr(self, name: str, lineno: int | None = None) -> nodes.ExtensionAttribute:
        return nodes.ExtensionAttribute(self.identifier, name, lineno=lineno)

    def call_method(
        self,
        name: str,
        args: list[nodes.Expr] | None = None,
        kwargs: list[nodes.Keyword] | None = None,
        dyn_args: nodes.Expr | None = None,
        dyn_kwargs: nodes.Expr | None = None,
        lineno: int | None = None,
    ) -> nodes.Call:
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        return nodes.Call(
            self.attr(name, lineno=lineno),
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=lineno,
        )


@pass_context
def _gettext_alias(
    __context: Context, *args: t.Any, **kwargs: t.Any
) -> t.Any | Undefined:
    return __context.call(__context.resolve("gettext"), *args, **kwargs)


def _make_new_gettext(func: t.Callable[[str], str]) -> t.Callable[..., str]:
    @pass_context
    def gettext(__context: Context, __string: str, **variables: t.Any) -> str:
        rv = __context.call(func, __string)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        return rv % variables  # type: ignore

    return gettext


def _make_new_ngettext(func: t.Callable[[str, str, int], str]) -> t.Callable[..., str]:
    @pass_context
    def ngettext(
        __context: Context,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("num", __num)
        rv = __context.call(func, __singular, __plural, __num)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        return rv % variables  # type: ignore

    return ngettext


def _make_new_pgettext(func: t.Callable[[str, str], str]) -> t.Callable[..., str]:
    @pass_context
    def pgettext(
        __context: Context, __string_ctx: str, __string: str, **variables: t.Any
    ) -> str:
        variables.setdefault("context", __string_ctx)
        rv = __context.call(func, __string_ctx, __string)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        return rv % variables  # type: ignore

    return pgettext


def _make_new_npgettext(
    func: t.Callable[[str, str, str, int], str],
) -> t.Callable[..., str]:
    @pass_context
    def npgettext(
        __context: Context,
        __string_ctx: str,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("context", __string_ctx)
        variables.setdefault("num", __num)
        rv = __context.call(func, __string_ctx, __singular, __plural, __num)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        return rv % variables  # type: ignore

    return npgettext


class InternationalizationExtension(Extension):
    tags = {"trans"}

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals["_"] = _gettext_alias
        environment.extend(
            install_gettext_translations=self._install,
            install_null_translations=self._install_null,
            install_gettext_callables=self._install_callables,
            uninstall_gettext_translations=self._uninstall,
            extract_translations=self._extract,
            newstyle_gettext=False,
        )

    def _install(
        self, translations: "_SupportedTranslations", newstyle: bool | None = None
    ) -> None:
        gettext = getattr(translations, "ugettext", None)
        if gettext is None:
            gettext = translations.gettext
        ngettext = getattr(translations, "ungettext", None)
        if ngettext is None:
            ngettext = translations.ngettext

        pgettext = getattr(translations, "pgettext", None)
        npgettext = getattr(translations, "npgettext", None)
        self._install_callables(
            gettext, ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext
        )

    def _install_null(self, newstyle: bool | None = None) -> None:
        import gettext

        translations = gettext.NullTranslations()
        self._install_callables(
            gettext=translations.gettext,
            ngettext=translations.ngettext,
            newstyle=newstyle,
            pgettext=translations.pgettext,
            npgettext=translations.npgettext,
        )

    def _install_callables(
        self,
        gettext: t.Callable[[str], str],
        ngettext: t.Callable[[str, str, int], str],
        newstyle: bool | None = None,
        pgettext: t.Callable[[str, str], str] | None = None,
        npgettext: t.Callable[[str, str, str, int], str] | None = None,
    ) -> None:
        env = self.environment
        if newstyle is not None:
            env.newstyle_gettext = newstyle  # type: ignore
        if env.newstyle_gettext:  # type: ignore
            gettext = _make_new_gettext(gettext)
            ngettext = _make_new_ngettext(ngettext)

            if pgettext is not None:
                pgettext = _make_new_pgettext(pgettext)

            if npgettext is not None:
                npgettext = _make_new_npgettext(npgettext)

        env.globals.update(
            gettext=gettext, ngettext=ngettext, pgettext=pgettext, npgettext=npgettext
        )

    def _uninstall(self, translations: "_SupportedTranslations") -> None:
        env_globals = self.environment.globals
        for key in ("gettext", "ngettext", "pgettext", "npgettext"):
            env_globals.pop(key, None)

    def _extract(
        self,
        source: str | nodes.Template,
        gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    ) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...]]]:
        if isinstance(source, str):
            source = self.environment.parse(source)
        return extract_from_ast(source, gettext_functions)

    def parse(self, parser: "Parser") -> nodes.Node | list[nodes.Node]:
        lineno = next(parser.stream).lineno

        context = None
        context_token = parser.stream.next_if("string")

        if context_token is not None:
            context = context_token.value

        plural_expr: nodes.Expr | None = None
        plural_expr_assignment: nodes.Assign | None = None
        num_called_num = False
        variables: dict[str, nodes.Expr] = {}
        trimmed = None
        stream = parser.stream
        expect = stream.expect
        skip_if = stream.skip_if
        while stream.current.type != "block_end":
            if variables:
                expect("comma")

            if skip_if("colon"):
                break

            token = expect("name")
            val = token.value
            if val in variables:
                parser.fail(
                    f"translatable variable {val!r} defined twice.",
                    token.lineno,
                    exc=TemplateAssertionError,
                )

            if stream.current.type == "assign":
                next(stream)
                variables[val] = var = parser.parse_expression()
            elif trimmed is None and val in ("trimmed", "notrimmed"):
                trimmed = val == "trimmed"
                continue
            else:
                variables[val] = var = nodes.Name(val, "load")

            if plural_expr is None:
                if isinstance(var, nodes.Call):
                    plural_expr = nodes.Name("_trans", "load")
                    variables[val] = plural_expr
                    plural_expr_assignment = nodes.Assign(
                        nodes.Name("_trans", "store"), var
                    )
                else:
                    plural_expr = var
                num_called_num = val == "num"

        expect("block_end")

        plural = None
        have_plural = False
        referenced = set()

        singular_names, singular = self._parse_block(parser, True)
        if singular_names:
            referenced.update(singular_names)
            if plural_expr is None:
                name0 = singular_names[0]
                plural_expr = nodes.Name(name0, "load")
                num_called_num = name0 == "num"

        test_pluralize = stream.current.test("name:pluralize")
        if test_pluralize:
            have_plural = True
            next(stream)
            if stream.current.type != "block_end":
                token = expect("name")
                tval = token.value
                if tval not in variables:
                    parser.fail(
                        f"unknown variable {tval!r} for pluralization",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )
                plural_expr = variables[tval]
                num_called_num = tval == "num"
            expect("block_end")
            plural_names, plural = self._parse_block(parser, False)
            next(stream)
            referenced.update(plural_names)
        else:
            next(stream)

        for name in referenced:
            if name not in variables:
                variables[name] = nodes.Name(name, "load")

        if not have_plural:
            plural_expr = None
        elif plural_expr is None:
            parser.fail("pluralize without variables", lineno)

        if trimmed is None:
            trimmed = self.environment.policies["ext.i18n.trimmed"]
        if trimmed:
            singular = self._trim_whitespace(singular)
            if plural:
                plural = self._trim_whitespace(plural)

        node = self._make_node(
            singular,
            plural,
            context,
            variables,
            plural_expr,
            bool(referenced),
            num_called_num and have_plural,
        )
        node.set_lineno(lineno)
        if plural_expr_assignment is not None:
            return [plural_expr_assignment, node]
        else:
            return node

    def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str] = _ws_re) -> str:
        return _ws_re.sub(" ", string.strip())

    def _parse_block(
        self, parser: "Parser", allow_pluralize: bool
    ) -> tuple[list[str], str]:
        referenced = []
        buf = []
        stream = parser.stream
        current = stream.current
        expect = stream.expect
        next_stream = next

        while True:
            ctype = stream.current.type
            if ctype == "data":
                buf.append(stream.current.value.replace("%", "%%"))
                next_stream(stream)
            elif ctype == "variable_begin":
                next_stream(stream)
                name = expect("name").value
                referenced.append(name)
                buf.append(f"%({name})s")
                expect("variable_end")
            elif ctype == "block_begin":
                next_stream(stream)
                current = stream.current
                block_name = current.value if current.type == "name" else None
                if block_name == "endtrans":
                    break
                elif block_name == "pluralize":
                    if allow_pluralize:
                        break
                    parser.fail(
                        "a translatable section can have only one pluralize section"
                    )
                elif block_name == "trans":
                    parser.fail(
                        "trans blocks can't be nested; did you mean `endtrans`?"
                    )
                parser.fail(
                    f"control structures in translatable sections are not allowed; "
                    f"saw `{block_name}`"
                )
            elif ctype == "eos":
                parser.fail("unclosed translation block")
            else:
                raise RuntimeError("internal parser error")

        return referenced, concat(buf)

    def _make_node(
        self,
        singular: str,
        plural: str | None,
        context: str | None,
        variables: dict[str, nodes.Expr],
        plural_expr: nodes.Expr | None,
        vars_referenced: bool,
        num_called_num: bool,
    ) -> nodes.Output:
        env = self.environment
        newstyle = env.newstyle_gettext  # type: ignore
        node: nodes.Expr

        if not vars_referenced and not newstyle:
            singular = singular.replace("%%", "%")
            if plural:
                plural = plural.replace("%%", "%")

        func_name = "gettext"
        func_args: list[nodes.Expr] = [nodes.Const(singular)]

        if context is not None:
            func_args.insert(0, nodes.Const(context))
            func_name = f"p{func_name}"

        if plural_expr is not None:
            func_name = f"n{func_name}"
            func_args.extend((nodes.Const(plural), plural_expr))

        node = nodes.Call(nodes.Name(func_name, "load"), func_args, [], None, None)

        if newstyle:
            node_kwargs = node.kwargs
            for key, value in variables.items():
                if num_called_num and key == "num":
                    continue
                node_kwargs.append(nodes.Keyword(key, value))

        else:
            node = nodes.MarkSafeIfAutoescape(node)
            if variables:
                n_dict = nodes.Dict(
                    [
                        nodes.Pair(nodes.Const(key), value)
                        for key, value in variables.items()
                    ]
                )
                node = nodes.Mod(node, n_dict)
        return nodes.Output([node])


class ExprStmtExtension(Extension):
    tags = {"do"}

    def parse(self, parser: "Parser") -> nodes.ExprStmt:
        node = nodes.ExprStmt(lineno=next(parser.stream).lineno)
        node.node = parser.parse_tuple()
        return node


class LoopControlExtension(Extension):
    tags = {"break", "continue"}

    def parse(self, parser: "Parser") -> nodes.Break | nodes.Continue:
        token = next(parser.stream)
        tval = token.value
        if tval == "break":
            return nodes.Break(lineno=token.lineno)
        return nodes.Continue(lineno=token.lineno)


class DebugExtension(Extension):
    tags = {"debug"}

    def parse(self, parser: "Parser") -> nodes.Output:
        lineno = parser.stream.expect("name:debug").lineno
        context = nodes.ContextReference()
        result = self.call_method("_render", [context], lineno=lineno)
        return nodes.Output([result], lineno=lineno)

    def _render(self, context: Context) -> str:
        filters_keys = self.environment.filters.keys()
        tests_keys = self.environment.tests.keys()
        result = {
            "context": context.get_all(),
            "filters": sorted(filters_keys),
            "tests": sorted(tests_keys),
        }
        return pprint.pformat(result, depth=3, compact=True)


def extract_from_ast(
    ast: nodes.Template,
    gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    babel_style: bool = True,
) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...]]]:
    out: str | None | tuple[str | None, ...]
    gfset = set(gettext_functions)
    find_all = ast.find_all
    nodes_Name = nodes.Name
    nodes_Const = nodes.Const

    for node in find_all(nodes.Call):
        node_node = node.node
        if (
            not isinstance(node_node, nodes_Name)
            or node_node.name not in gfset
        ):
            continue

        strings: list[str | None] = []

        for arg in node.args:
            if isinstance(arg, nodes_Const) and isinstance(arg.value, str):
                strings.append(arg.value)
            else:
                strings.append(None)

        strings_extend = strings.extend
        # For each keyword arg, each dyn_args/kwargs, we append None (as before)
        if node.kwargs:
            strings_extend([None] * len(node.kwargs))
        if node.dyn_args is not None:
            strings.append(None)
        if node.dyn_kwargs is not None:
            strings.append(None)

        if not babel_style:
            out = tuple(x for x in strings if x is not None)
            if not out:
                continue
        else:
            if len(strings) == 1:
                out = strings[0]
            else:
                out = tuple(strings)

        yield node.lineno, node_node.name, out


class _CommentFinder:
    def __init__(
        self, tokens: t.Sequence[tuple[int, str, str]], comment_tags: t.Sequence[str]
    ) -> None:
        self.tokens = tokens
        self.comment_tags = comment_tags
        self.offset = 0
        self.last_lineno = 0

    def find_backwards(self, offset: int) -> list[str]:
        toks = self.tokens
        ctags = self.comment_tags
        try:
            for _, token_type, token_value in reversed(
                toks[self.offset:offset]
            ):
                if token_type == "comment" or token_type == "linecomment":
                    try:
                        prefix, comment = token_value.split(None, 1)
                    except ValueError:
                        continue
                    if prefix in ctags:
                        return [comment.rstrip()]
            return []
        finally:
            self.offset = offset

    def find_comments(self, lineno: int) -> list[str]:
        if not self.comment_tags or self.last_lineno > lineno:
            return []
        toks = self.tokens
        off = self.offset
        for idx, (token_lineno, _, _) in enumerate(toks[off:]):
            if token_lineno > lineno:
                return self.find_backwards(off + idx)
        return self.find_backwards(len(toks))


def babel_extract(
    fileobj: t.BinaryIO,
    keywords: t.Sequence[str],
    comment_tags: t.Sequence[str],
    options: dict[str, t.Any],
) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...], list[str]]]:
    extensions: dict[type[Extension], None] = {}

    import_string_local = import_string  # localize for loop

    ext_names = options.get("extensions", "").split(",")
    for extension_name in ext_names:
        extension_name = extension_name.strip()
        if not extension_name:
            continue
        extensions[import_string_local(extension_name)] = None

    if InternationalizationExtension not in extensions:
        extensions[InternationalizationExtension] = None

    def getbool(options: t.Mapping[str, str], key: str, default: bool = False) -> bool:
        return options.get(key, str(default)).lower() in {"1", "on", "yes", "true"}

    silent = getbool(options, "silent", True)
    environment = Environment(
        options.get("block_start_string", defaults.BLOCK_START_STRING),
        options.get("block_end_string", defaults.BLOCK_END_STRING),
        options.get("variable_start_string", defaults.VARIABLE_START_STRING),
        options.get("variable_end_string", defaults.VARIABLE_END_STRING),
        options.get("comment_start_string", defaults.COMMENT_START_STRING),
        options.get("comment_end_string", defaults.COMMENT_END_STRING),
        options.get("line_statement_prefix") or defaults.LINE_STATEMENT_PREFIX,
        options.get("line_comment_prefix") or defaults.LINE_COMMENT_PREFIX,
        getbool(options, "trim_blocks", defaults.TRIM_BLOCKS),
        getbool(options, "lstrip_blocks", defaults.LSTRIP_BLOCKS),
        defaults.NEWLINE_SEQUENCE,
        getbool(options, "keep_trailing_newline", defaults.KEEP_TRAILING_NEWLINE),
        tuple(extensions),
        cache_size=0,
        auto_reload=False,
    )

    if getbool(options, "trimmed"):
        environment.policies["ext.i18n.trimmed"] = True
    if getbool(options, "newstyle_gettext"):
        environment.newstyle_gettext = True  # type: ignore

    source = fileobj.read().decode(options.get("encoding", "utf-8"))
    try:
        node = environment.parse(source)
        tokens = list(environment.lex(environment.preprocess(source)))
    except TemplateSyntaxError:
        if not silent:
            raise
        return

    finder = _CommentFinder(tokens, comment_tags)
    finder_findcomments = finder.find_comments  # cache lookup
    for lineno, func, message in extract_from_ast(node, keywords):
        yield lineno, func, message, finder_findcomments(lineno)


i18n = InternationalizationExtension
do = ExprStmtExtension
loopcontrols = LoopControlExtension
debug = DebugExtension