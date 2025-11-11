import sys
import typing as t
from types import CodeType
from types import TracebackType

from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing

if t.TYPE_CHECKING:
    from .runtime import Context


def rewrite_traceback_stack(source: str | None = None) -> BaseException:
    _, exc_value, tb = sys.exc_info()
    exc_value = t.cast(BaseException, exc_value)
    tb = t.cast(TracebackType, tb)

    if isinstance(exc_value, TemplateSyntaxError) and not exc_value.translated:
        exc_value.translated = True
        exc_value.source = source
        exc_value.with_traceback(None)
        tb = fake_traceback(
            exc_value, None, exc_value.filename or "<unknown>", exc_value.lineno
        )
    else:
        tb = tb.tb_next

    stack = []

    while tb is not None:
        tb_frame = tb.tb_frame
        tb_frame_f_code = tb_frame.f_code
        # Skip frames decorated with @internalcode.
        if tb_frame_f_code in internal_code:
            tb = tb.tb_next
            continue

        tb_frame_f_globals = tb_frame.f_globals
        template = tb_frame_f_globals.get("__jinja_template__")

        if template is not None:
            lineno = template.get_corresponding_lineno(tb.tb_lineno)
            fake_tb = fake_traceback(exc_value, tb, template.filename, lineno)
            stack.append(fake_tb)
        else:
            stack.append(tb)

        tb = tb.tb_next

    tb_next = None

    for tb in reversed(stack):
        tb.tb_next = tb_next
        tb_next = tb

    return exc_value.with_traceback(tb_next)


def fake_traceback(  # type: ignore
    exc_value: BaseException, tb: TracebackType | None, filename: str, lineno: int
) -> TracebackType:
    if tb is not None:
        locals_dict = get_template_locals(tb.tb_frame.f_locals)
        locals_dict.pop("__jinja_exception__", None)
    else:
        locals_dict = {}

    globals_dict = {
        "__name__": filename,
        "__file__": filename,
        "__jinja_exception__": exc_value,
    }
    code: CodeType = compile(
        "\n" * (lineno - 1) + "raise __jinja_exception__", filename, "exec"
    )

    location = "template"

    if tb is not None:
        function = tb.tb_frame.f_code.co_name

        if function == "root":
            location = "top-level template code"
        elif function.startswith("block_"):
            location = f"block {function[6:]!r}"

    code = code.replace(co_name=location)

    try:
        exec(code, globals_dict, locals_dict)
    except BaseException:
        return sys.exc_info()[2].tb_next  # type: ignore


def get_template_locals(real_locals: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
    ctx: "Context | None" = real_locals.get("context")

    if ctx is not None:
        data: dict[str, t.Any] = ctx.get_all().copy()
    else:
        data = {}

    local_overrides: dict[str, tuple[int, t.Any]] = {}

    real_locals_items = real_locals.items()
    missing_obj = missing  # lookup once

    for name, value in real_locals_items:
        if not name.startswith("l_") or value is missing_obj:
            continue

        try:
            _, depth_str, real_name = name.split("_", 2)
            depth = int(depth_str)
        except ValueError:
            continue

        cur_depth = local_overrides.get(real_name, (-1,))[0]

        if cur_depth < depth:
            local_overrides[real_name] = (depth, value)

    for name, (_, value) in local_overrides.items():
        if value is missing_obj:
            data.pop(name, None)
        else:
            data[name] = value

    return data