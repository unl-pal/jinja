"""Built-in template filters used with the ``|`` operator."""

import math
import random
import re
import typing
import typing as t
from collections import abc
from inspect import getattr_static
from itertools import chain
from itertools import groupby

from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str

from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment
    from .nodes import EvalContext
    from .runtime import Context
    from .sandbox import SandboxedEnvironment  # noqa: F401

    class HasHTML(te.Protocol):
        def __html__(self) -> str:
            pass


F = t.TypeVar("F", bound=t.Callable[..., t.Any])
K = t.TypeVar("K")
V = t.TypeVar("V")


def ignore_case(value: V) -> V:
    if isinstance(value, str):
        return t.cast(V, value.lower())
    return value


def make_attrgetter(
    environment: "Environment",
    attribute: str | int | None,
    postprocess: t.Callable[[t.Any], t.Any] | None = None,
    default: t.Any | None = None,
) -> t.Callable[[t.Any], t.Any]:
    parts = _prepare_attribute_parts(attribute)

    def attrgetter(item: t.Any) -> t.Any:
        for part in parts:
            item = environment.getitem(item, part)
            if default is not None and isinstance(item, Undefined):
                item = default
        if postprocess is not None:
            item = postprocess(item)
        return item

    return attrgetter


def make_multi_attrgetter(
    environment: "Environment",
    attribute: str | int | None,
    postprocess: t.Callable[[t.Any], t.Any] | None = None,
) -> t.Callable[[t.Any], list[t.Any]]:
    if isinstance(attribute, str):
        split: t.Sequence[str | int | None] = attribute.split(",")
    else:
        split = [attribute]
    parts = [_prepare_attribute_parts(item) for item in split]

    def attrgetter(item: t.Any) -> list[t.Any]:
        n = len(parts)
        items = [None] * n
        _getitem = environment.getitem
        _postprocess = postprocess
        for i, attribute_part in enumerate(parts):
            item_i = item
            for part in attribute_part:
                item_i = _getitem(item_i, part)
            if _postprocess is not None:
                item_i = _postprocess(item_i)
            items[i] = item_i
        return items

    return attrgetter


def _prepare_attribute_parts(
    attr: str | int | None,
) -> list[str | int]:
    if attr is None:
        return []
    if isinstance(attr, str):
        return [int(x) if x.isdigit() else x for x in attr.split(".")]
    return [attr]


def do_forceescape(value: "str | HasHTML") -> Markup:
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()
    return escape(str(value))


def do_urlencode(
    value: str | t.Mapping[str, t.Any] | t.Iterable[tuple[str, t.Any]],
) -> str:
    if isinstance(value, str) or not isinstance(value, abc.Iterable):
        return url_quote(value)
    if isinstance(value, dict):
        items: t.Iterable[tuple[str, t.Any]] = value.items()
    else:
        items = value  # type: ignore
    url_q = url_quote
    return "&".join(
        f"{url_q(k, for_qs=True)}={url_q(v, for_qs=True)}" for k, v in items
    )


@pass_eval_context
def do_replace(
    eval_ctx: "EvalContext", s: str, old: str, new: str, count: int | None = None
) -> str:
    if count is None:
        count = -1
    if not eval_ctx.autoescape:
        return str(s).replace(str(old), str(new), count)
    old_html = hasattr(old, "__html__")
    new_html = hasattr(new, "__html__")
    s_html = hasattr(s, "__html__")
    if (old_html or new_html) and not s_html:
        s = escape(s)
    else:
        s = soft_str(s)
    return s.replace(soft_str(old), soft_str(new), count)


def do_upper(s: str) -> str:
    return soft_str(s).upper()


def do_lower(s: str) -> str:
    return soft_str(s).lower()


def do_items(value: t.Mapping[K, V] | Undefined) -> t.Iterator[tuple[K, V]]:
    if isinstance(value, Undefined):
        return
    if not isinstance(value, abc.Mapping):
        raise TypeError("Can only get item pairs from a mapping.")
    yield from value.items()


_attr_key_re = re.compile(r"[\s/>=]", flags=re.ASCII)


@pass_eval_context
def do_xmlattr(
    eval_ctx: "EvalContext", d: t.Mapping[str, t.Any], autospace: bool = True
) -> str:
    items = []
    _attr_search = _attr_key_re.search
    _escape = escape
    for key, value in d.items():
        if value is None or isinstance(value, Undefined):
            continue
        if _attr_search(key) is not None:
            raise ValueError(f"Invalid character in attribute name: {key!r}")
        items.append(f'{_escape(key)}="{_escape(value)}"')
    rv = " ".join(items)
    if autospace and rv:
        rv = " " + rv
    if eval_ctx.autoescape:
        rv = Markup(rv)
    return rv


def do_capitalize(s: str) -> str:
    return soft_str(s).capitalize()


_word_beginning_split_re = re.compile(r"([-\s({\[<]+)")


def do_title(s: str) -> str:
    parts = _word_beginning_split_re.split(soft_str(s))
    # Avoid list overhead by appending as we go
    res = []
    append = res.append
    for item in parts:
        if item:
            append(item[0].upper() + item[1:].lower())
    return "".join(res)


def do_dictsort(
    value: t.Mapping[K, V],
    case_sensitive: bool = False,
    by: 'te.Literal["key", "value"]' = "key",
    reverse: bool = False,
) -> list[tuple[K, V]]:
    if by == "key":
        pos = 0
    elif by == "value":
        pos = 1
    else:
        raise FilterArgumentError('You can only sort by either "key" or "value"')
    def sort_func(item: tuple[t.Any, t.Any]) -> t.Any:
        value_or_key = item[pos]
        if not case_sensitive:
            value_or_key = ignore_case(value_or_key)
        return value_or_key
    return sorted(value.items(), key=sort_func, reverse=reverse)


@pass_environment
def do_sort(
    environment: "Environment",
    value: "t.Iterable[V]",
    reverse: bool = False,
    case_sensitive: bool = False,
    attribute: str | int | None = None,
) -> "list[V]":
    key_func = make_multi_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return sorted(value, key=key_func, reverse=reverse)


@pass_environment
def sync_do_unique(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: str | int | None = None,
) -> "t.Iterator[V]":
    getter = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    seen = set()
    add = seen.add
    for item in value:
        key = getter(item)
        if key not in seen:
            add(key)
            yield item


@async_variant(sync_do_unique)  # type: ignore
async def do_unique(
    environment: "Environment",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: str | int | None = None,
) -> "t.Iterator[V]":
    return sync_do_unique(
        environment, await auto_to_list(value), case_sensitive, attribute
    )


def _min_or_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    func: "t.Callable[..., V]",
    case_sensitive: bool,
    attribute: str | int | None,
) -> "V | Undefined":
    it = iter(value)
    try:
        first = next(it)
    except StopIteration:
        return environment.undefined("No aggregated item, sequence was empty.")
    key_func = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return func(chain([first], it), key=key_func)


@pass_environment
def do_min(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: str | int | None = None,
) -> "V | Undefined":
    return _min_or_max(environment, value, min, case_sensitive, attribute)


@pass_environment
def do_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: str | int | None = None,
) -> "V | Undefined":
    return _min_or_max(environment, value, max, case_sensitive, attribute)


def do_default(
    value: V,
    default_value: V = "",  # type: ignore
    boolean: bool = False,
) -> V:
    if isinstance(value, Undefined) or (boolean and not value):
        return default_value
    return value


@pass_eval_context
def sync_do_join(
    eval_ctx: "EvalContext",
    value: t.Iterable[t.Any],
    d: str = "",
    attribute: str | int | None = None,
) -> str:
    if attribute is not None:
        value = map(make_attrgetter(eval_ctx.environment, attribute), value)
    if not eval_ctx.autoescape:
        return str(d).join(map(str, value))
    if not hasattr(d, "__html__"):
        value = list(value)
        do_escape = False
        for idx, item in enumerate(value):
            if hasattr(item, "__html__"):
                do_escape = True
            else:
                value[idx] = str(item)
        if do_escape:
            d = escape(d)
        else:
            d = str(d)
        return d.join(value)
    return soft_str(d).join(map(soft_str, value))


@async_variant(sync_do_join)  # type: ignore
async def do_join(
    eval_ctx: "EvalContext",
    value: t.AsyncIterable[t.Any] | t.Iterable[t.Any],
    d: str = "",
    attribute: str | int | None = None,
) -> str:
    return sync_do_join(eval_ctx, await auto_to_list(value), d, attribute)


def do_center(value: str, width: int = 80) -> str:
    return soft_str(value).center(width)


@pass_environment
def sync_do_first(environment: "Environment", seq: "t.Iterable[V]") -> "V | Undefined":
    try:
        return next(iter(seq))
    except StopIteration:
        return environment.undefined("No first item, sequence was empty.")


@async_variant(sync_do_first)  # type: ignore
async def do_first(
    environment: "Environment", seq: "t.AsyncIterable[V] | t.Iterable[V]"
) -> "V | Undefined":
    try:
        return await auto_aiter(seq).__anext__()
    except StopAsyncIteration:
        return environment.undefined("No first item, sequence was empty.")


@pass_environment
def do_last(environment: "Environment", seq: "t.Reversible[V]") -> "V | Undefined":
    try:
        return next(iter(reversed(seq)))
    except StopIteration:
        return environment.undefined("No last item, sequence was empty.")


@pass_context
def do_random(context: "Context", seq: "t.Sequence[V]") -> "V | Undefined":
    try:
        return random.choice(seq)
    except IndexError:
        return context.environment.undefined("No random item, sequence was empty.")


def do_filesizeformat(value: str | float | int, binary: bool = False) -> str:
    bytes = float(value)
    base = 1024 if binary else 1000
    prefixes = [
        ("KiB" if binary else "kB"),
        ("MiB" if binary else "MB"),
        ("GiB" if binary else "GB"),
        ("TiB" if binary else "TB"),
        ("PiB" if binary else "PB"),
        ("EiB" if binary else "EB"),
        ("ZiB" if binary else "ZB"),
        ("YiB" if binary else "YB"),
    ]
    if bytes == 1:
        return "1 Byte"
    elif bytes < base:
        return f"{int(bytes)} Bytes"
    else:
        for i, prefix in enumerate(prefixes):
            unit = base ** (i + 2)
            if bytes < unit:
                return f"{base * bytes / unit:.1f} {prefix}"
        return f"{base * bytes / unit:.1f} {prefix}"


def do_pprint(value: t.Any) -> str:
    return pformat(value)


_uri_scheme_re = re.compile(r"^([\w.+-]{2,}:(/){0,2})$")


@pass_eval_context
def do_urlize(
    eval_ctx: "EvalContext",
    value: str,
    trim_url_limit: int | None = None,
    nofollow: bool = False,
    target: str | None = None,
    rel: str | None = None,
    extra_schemes: t.Iterable[str] | None = None,
) -> str:
    policies = eval_ctx.environment.policies
    rel_parts = set((rel or "").split())
    if nofollow:
        rel_parts.add("nofollow")
    rel_parts.update((policies["urlize.rel"] or "").split())
    rel = " ".join(sorted(rel_parts)) or None
    if target is None:
        target = policies["urlize.target"]
    if extra_schemes is None:
        extra_schemes = policies["urlize.extra_schemes"] or ()
    uri_scheme_fullmatch = _uri_scheme_re.fullmatch
    for scheme in extra_schemes:
        if uri_scheme_fullmatch(scheme) is None:
            raise FilterArgumentError(f"{scheme!r} is not a valid URI scheme prefix.")
    rv = urlize(
        value,
        trim_url_limit=trim_url_limit,
        rel=rel,
        target=target,
        extra_schemes=extra_schemes,
    )
    if eval_ctx.autoescape:
        rv = Markup(rv)
    return rv


def do_indent(
    s: str, width: int | str = 4, first: bool = False, blank: bool = False
) -> str:
    if isinstance(width, str):
        indention = width
    else:
        indention = " " * width
    newline = "\n"
    if isinstance(s, Markup):
        indention = Markup(indention)
        newline = Markup(newline)
    s += newline  # this quirk is necessary for splitlines method
    if blank:
        rv = (newline + indention).join(s.splitlines())
    else:
        lines = s.splitlines()
        rv = lines.pop(0)
        if lines:
            join_line = newline.join
            rv += newline + join_line(
                indention + line if line else line for line in lines
            )
    if first:
        rv = indention + rv
    return rv


@pass_environment
def do_truncate(
    env: "Environment",
    s: str,
    length: int = 255,
    killwords: bool = False,
    end: str = "...",
    leeway: int | None = None,
) -> str:
    if leeway is None:
        leeway = env.policies["truncate.leeway"]
    assert length >= len(end), f"expected length >= {len(end)}, got {length}"
    assert leeway >= 0, f"expected leeway >= 0, got {leeway}"
    if len(s) <= length + leeway:
        return s
    if killwords:
        return s[: length - len(end)] + end
    result = s[: length - len(end)].rsplit(" ", 1)[0]
    return result + end


@pass_environment
def do_wordwrap(
    environment: "Environment",
    s: str,
    width: int = 79,
    break_long_words: bool = True,
    wrapstring: str | None = None,
    break_on_hyphens: bool = True,
) -> str:
    import textwrap

    if wrapstring is None:
        wrapstring = environment.newline_sequence

    wrap = textwrap.wrap
    join_wrap = wrapstring.join
    join_line = wrapstring.join
    return join_wrap(
        [
            join_line(
                wrap(
                    line,
                    width=width,
                    expand_tabs=False,
                    replace_whitespace=False,
                    break_long_words=break_long_words,
                    break_on_hyphens=break_on_hyphens,
                )
            )
            for line in s.splitlines()
        ]
    )


_word_re = re.compile(r"\w+")


def do_wordcount(s: str) -> int:
    return len(_word_re.findall(soft_str(s)))


def do_int(value: t.Any, default: int = 0, base: int = 10) -> int:
    try:
        if isinstance(value, str):
            return int(value, base)
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def do_float(value: t.Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def do_format(value: str, *args: t.Any, **kwargs: t.Any) -> str:
    if args and kwargs:
        raise FilterArgumentError(
            "can't handle positional and keyword arguments at the same time"
        )
    return soft_str(value) % (kwargs or args)


def do_trim(value: str, chars: str | None = None) -> str:
    return soft_str(value).strip(chars)


def do_striptags(value: "str | HasHTML") -> str:
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()
    return Markup(str(value)).striptags()


def sync_do_slice(
    value: "t.Collection[V]", slices: int, fill_with: "V | None" = None
) -> "t.Iterator[list[V]]":
    seq = list(value)
    length = len(seq)
    items_per_slice = length // slices
    slices_with_extra = length % slices
    offset = 0
    for slice_number in range(slices):
        start = offset + slice_number * items_per_slice
        if slice_number < slices_with_extra:
            offset += 1
        end = offset + (slice_number + 1) * items_per_slice
        tmp = seq[start:end]
        if fill_with is not None and slice_number >= slices_with_extra:
            tmp.append(fill_with)
        yield tmp


@async_variant(sync_do_slice)  # type: ignore
async def do_slice(
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    slices: int,
    fill_with: t.Any | None = None,
) -> "t.Iterator[list[V]]":
    return sync_do_slice(await auto_to_list(value), slices, fill_with)


def do_batch(
    value: "t.Iterable[V]", linecount: int, fill_with: "V | None" = None
) -> "t.Iterator[list[V]]":
    tmp: list[V] = []
    for item in value:
        if len(tmp) == linecount:
            yield tmp
            tmp = []
        tmp.append(item)
    ltmp = len(tmp)
    if tmp:
        if fill_with is not None and ltmp < linecount:
            tmp += [fill_with] * (linecount - ltmp)
        yield tmp


def do_round(
    value: float,
    precision: int = 0,
    method: 'te.Literal["common", "ceil", "floor"]' = "common",
) -> float:
    if method not in {"common", "ceil", "floor"}:
        raise FilterArgumentError("method must be common, ceil or floor")
    if method == "common":
        return round(value, precision)
    func = getattr(math, method)
    return t.cast(float, func(value * (10**precision)) / (10**precision))


class _GroupTuple(t.NamedTuple):
    grouper: t.Any
    list: list[t.Any]

    def __repr__(self) -> str:
        return tuple.__repr__(self)

    def __str__(self) -> str:
        return tuple.__str__(self)


@pass_environment
def sync_do_groupby(
    environment: "Environment",
    value: "t.Iterable[V]",
    attribute: str | int,
    default: t.Any | None = None,
    case_sensitive: bool = False,
) -> "list[_GroupTuple]":
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    sorted_val = sorted(value, key=expr)
    out = [
        _GroupTuple(key, list(values))
        for key, values in groupby(sorted_val, expr)
    ]
    if not case_sensitive:
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]
    return out


@async_variant(sync_do_groupby)  # type: ignore
async def do_groupby(
    environment: "Environment",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    attribute: str | int,
    default: t.Any | None = None,
    case_sensitive: bool = False,
) -> "list[_GroupTuple]":
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    vals = await auto_to_list(value)
    sorted_val = sorted(vals, key=expr)
    out = [
        _GroupTuple(key, await auto_to_list(values))
        for key, values in groupby(sorted_val, expr)
    ]
    if not case_sensitive:
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]
    return out


@pass_environment
def sync_do_sum(
    environment: "Environment",
    iterable: "t.Iterable[V]",
    attribute: str | int | None = None,
    start: V = 0,  # type: ignore
) -> V:
    if attribute is not None:
        iterable = map(make_attrgetter(environment, attribute), iterable)
    return sum(iterable, start)  # type: ignore[no-any-return, call-overload]


@async_variant(sync_do_sum)  # type: ignore
async def do_sum(
    environment: "Environment",
    iterable: "t.AsyncIterable[V] | t.Iterable[V]",
    attribute: str | int | None = None,
    start: V = 0,  # type: ignore
) -> V:
    rv = start
    if attribute is not None:
        func = make_attrgetter(environment, attribute)
    else:
        def func(x: V) -> V:
            return x
    async for item in auto_aiter(iterable):
        rv += func(item)
    return rv


def sync_do_list(value: "t.Iterable[V]") -> "list[V]":
    return list(value)


@async_variant(sync_do_list)  # type: ignore
async def do_list(value: "t.AsyncIterable[V] | t.Iterable[V]") -> "list[V]":
    return await auto_to_list(value)


def do_mark_safe(value: str) -> Markup:
    return Markup(value)


def do_mark_unsafe(value: str) -> str:
    return str(value)


@typing.overload
def do_reverse(value: str) -> str: ...


@typing.overload
def do_reverse(value: "t.Iterable[V]") -> "t.Iterable[V]": ...


def do_reverse(value: str | t.Iterable[V]) -> str | t.Iterable[V]:
    if isinstance(value, str):
        return value[::-1]
    try:
        return reversed(value)  # type: ignore
    except TypeError:
        try:
            rv = list(value)
            rv.reverse()
            return rv
        except TypeError as e:
            raise FilterArgumentError("argument must be iterable") from e


@pass_environment
def do_attr(environment: "Environment", obj: t.Any, name: str) -> Undefined | t.Any:
    try:
        getattr_static(obj, name)
    except AttributeError:
        if not hasattr(obj, name):
            return environment.undefined(obj=obj, name=name)
    return environment.getattr(obj, name)


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    *,
    attribute: str = ...,
    default: t.Any | None = None,
) -> t.Iterable[t.Any]: ...


@pass_context
def sync_do_map(
    context: "Context", value: t.Iterable[t.Any], *args: t.Any, **kwargs: t.Any
) -> t.Iterable[t.Any]:
    if value:
        func = prepare_map(context, args, kwargs)
        for item in value:
            yield func(item)


@typing.overload
def do_map(
    context: "Context",
    value: t.AsyncIterable[t.Any] | t.Iterable[t.Any],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def do_map(
    context: "Context",
    value: t.AsyncIterable[t.Any] | t.Iterable[t.Any],
    *,
    attribute: str = ...,
    default: t.Any | None = None,
) -> t.Iterable[t.Any]: ...


@async_variant(sync_do_map)  # type: ignore
async def do_map(
    context: "Context",
    value: t.AsyncIterable[t.Any] | t.Iterable[t.Any],
    *args: t.Any,
    **kwargs: t.Any,
) -> t.AsyncIterable[t.Any]:
    if value:
        func = prepare_map(context, args, kwargs)
        async for item in auto_aiter(value):
            yield await auto_await(func(item))


@pass_context
def sync_do_select(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    return select_or_reject(context, value, args, kwargs, lambda x: x, False)


@async_variant(sync_do_select)  # type: ignore
async def do_select(
    context: "Context",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, False)


@pass_context
def sync_do_reject(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    return select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@async_variant(sync_do_reject)  # type: ignore
async def do_reject(
    context: "Context",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@pass_context
def sync_do_selectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    return select_or_reject(context, value, args, kwargs, lambda x: x, True)


@async_variant(sync_do_selectattr)  # type: ignore
async def do_selectattr(
    context: "Context",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, True)


@pass_context
def sync_do_rejectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    return select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@async_variant(sync_do_rejectattr)  # type: ignore
async def do_rejectattr(
    context: "Context",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@pass_eval_context
def do_tojson(
    eval_ctx: "EvalContext", value: t.Any, indent: int | None = None
) -> Markup:
    policies = eval_ctx.environment.policies
    dumps = policies["json.dumps_function"]
    kwargs = policies["json.dumps_kwargs"]
    if indent is not None:
        kwargs = kwargs.copy()
        kwargs["indent"] = indent
    return htmlsafe_json_dumps(value, dumps=dumps, **kwargs)


def prepare_map(
    context: "Context", args: tuple[t.Any, ...], kwargs: dict[str, t.Any]
) -> t.Callable[[t.Any], t.Any]:
    if not args and "attribute" in kwargs:
        attribute = kwargs.pop("attribute")
        default = kwargs.pop("default", None)
        if kwargs:
            raise FilterArgumentError(
                f"Unexpected keyword argument {next(iter(kwargs))!r}"
            )
        func = make_attrgetter(context.environment, attribute, default=default)
    else:
        try:
            name = args[0]
            args = args[1:]
        except LookupError:
            raise FilterArgumentError("map requires a filter argument") from None

        def func(item: t.Any) -> t.Any:
            return context.environment.call_filter(
                name, item, args, kwargs, context=context
            )

    return func


def prepare_select_or_reject(
    context: "Context",
    args: tuple[t.Any, ...],
    kwargs: dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> t.Callable[[t.Any], t.Any]:
    if lookup_attr:
        try:
            attr = args[0]
        except LookupError:
            raise FilterArgumentError("Missing parameter for attribute name") from None

        transfunc = make_attrgetter(context.environment, attr)
        off = 1
    else:
        off = 0

        def transfunc(x: V) -> V:
            return x

    try:
        name = args[off]
        args = args[1 + off :]

        def func(item: t.Any) -> t.Any:
            return context.environment.call_test(name, item, args, kwargs, context)

    except LookupError:
        func = bool  # type: ignore

    return lambda item: modfunc(func(transfunc(item)))


def select_or_reject(
    context: "Context",
    value: "t.Iterable[V]",
    args: tuple[t.Any, ...],
    kwargs: dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.Iterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)
        for item in value:
            if func(item):
                yield item


async def async_select_or_reject(
    context: "Context",
    value: "t.AsyncIterable[V] | t.Iterable[V]",
    args: tuple[t.Any, ...],
    kwargs: dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.AsyncIterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)
        async for item in auto_aiter(value):
            if func(item):
                yield item


FILTERS = {
    "abs": abs,
    "attr": do_attr,
    "batch": do_batch,
    "capitalize": do_capitalize,
    "center": do_center,
    "count": len,
    "d": do_default,
    "default": do_default,
    "dictsort": do_dictsort,
    "e": escape,
    "escape": escape,
    "filesizeformat": do_filesizeformat,
    "first": do_first,
    "float": do_float,
    "forceescape": do_forceescape,
    "format": do_format,
    "groupby": do_groupby,
    "indent": do_indent,
    "int": do_int,
    "join": do_join,
    "last": do_last,
    "length": len,
    "list": do_list,
    "lower": do_lower,
    "items": do_items,
    "map": do_map,
    "min": do_min,
    "max": do_max,
    "pprint": do_pprint,
    "random": do_random,
    "reject": do_reject,
    "rejectattr": do_rejectattr,
    "replace": do_replace,
    "reverse": do_reverse,
    "round": do_round,
    "safe": do_mark_safe,
    "select": do_select,
    "selectattr": do_selectattr,
    "slice": do_slice,
    "sort": do_sort,
    "string": soft_str,
    "striptags": do_striptags,
    "sum": do_sum,
    "title": do_title,
    "trim": do_trim,
    "truncate": do_truncate,
    "unique": do_unique,
    "upper": do_upper,
    "urlencode": do_urlencode,
    "urlize": do_urlize,
    "wordcount": do_wordcount,
    "wordwrap": do_wordwrap,
    "xmlattr": do_xmlattr,
    "tojson": do_tojson,
}