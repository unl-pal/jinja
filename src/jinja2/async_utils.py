import inspect
import typing as t
from functools import WRAPPER_ASSIGNMENTS
from functools import wraps

from .utils import _PassArg
from .utils import pass_eval_context

if t.TYPE_CHECKING:
    import typing_extensions as te

V = t.TypeVar("V")

_async_func_attrs = ("__module__", "__name__", "__qualname__")
_normal_func_attrs = tuple(set(WRAPPER_ASSIGNMENTS).difference(_async_func_attrs))
_common_primitives = {int, float, bool, str, list, dict, tuple, type(None)}
_type = type
_inspect_isawaitable = inspect.isawaitable

def async_variant(normal_func):  # type: ignore
    def decorator(async_func):  # type: ignore
        pass_arg = _PassArg.from_obj(normal_func)
        need_eval_context = pass_arg is None

        environment_sentinel = _PassArg.environment
        cast = t.cast

        if pass_arg is environment_sentinel:
            def is_async(args: t.Any) -> bool:
                return cast(bool, args[0].is_async)
        else:
            def is_async(args: t.Any) -> bool:
                # Use local variable to minimize attribute chain lookup
                env = args[0].environment
                return cast(bool, env.is_async)

        @wraps(normal_func, assigned=_normal_func_attrs)
        @wraps(async_func, assigned=_async_func_attrs, updated=())
        def wrapper(*args, **kwargs):  # type: ignore
            b = is_async(args)

            if need_eval_context:
                args = args[1:]

            if b:
                return async_func(*args, **kwargs)

            return normal_func(*args, **kwargs)

        if need_eval_context:
            wrapper = pass_eval_context(wrapper)

        wrapper.jinja_async_variant = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


async def auto_await(value: t.Union[t.Awaitable["V"], "V"]) -> "V":
    # Avoid a costly call to isawaitable
    if _type(value) in _common_primitives:
        return t.cast("V", value)

    if _inspect_isawaitable(value):
        return await t.cast("t.Awaitable[V]", value)

    return value


class _IteratorToAsyncIterator(t.Generic[V]):
    def __init__(self, iterator: "t.Iterator[V]"):
        self._iterator = iterator

    def __aiter__(self) -> "te.Self":
        return self

    async def __anext__(self) -> V:
        # Use local variable for performance
        iterator = self._iterator
        try:
            return next(iterator)
        except StopIteration as e:
            raise StopAsyncIteration(e.value) from e


def auto_aiter(
    iterable: "t.AsyncIterable[V] | t.Iterable[V]",
) -> "t.AsyncIterator[V]":
    # Hoist method lookup
    aiter = getattr(iterable, "__aiter__", None)
    if aiter is not None:
        return aiter()
    else:
        return _IteratorToAsyncIterator(iter(iterable))


async def auto_to_list(
    value: "t.AsyncIterable[V] | t.Iterable[V]",
) -> list["V"]:
    return [x async for x in auto_aiter(value)]