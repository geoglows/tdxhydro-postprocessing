"""
Terminal output shared by the pipeline steps.

Most steps log to a file under $RFS_DATA_ROOT/hydrography-scratchfiles/logs and print almost
nothing, which is right for the detail but leaves the terminal running pipeline.sh silent for
long stretches - steps 3 and 4 in particular run for the better part of an hour with no stdout at
all. These helpers are the other half of that: a marker on stdout saying which step is starting,
and a summary at the end of the release saying what came out.
"""
WIDTH = 78


def banner(title: str) -> None:
    """Print the dashed marker that opens a step, naming the step and nothing else.

    Deliberately no position in the sequence: "step 4 of 6" and "after step 5" are facts about an
    ordering that has already changed more than once here - steps have been merged, retired and
    renumbered - and a hardcoded one goes quietly wrong rather than failing. The name says which
    step this is, which is the part that stays true.

    Assembled into a single string and written in one ``print`` on purpose: a banner printed a
    line at a time interleaves with a neighbouring process's. One write of ~240 bytes does not.
    """
    rule = '-' * WIDTH
    print(f'\n{rule}\n{title}\n{rule}', flush=True)


def summary(title: str, rows, notes=()) -> None:
    """Print a dashed block of ``label: value`` rows, values right-aligned in one column.

    ``rows`` is an iterable of ``(label, value)`` pairs; values are printed as given, so the
    caller owns the formatting (thousands separators, units). ``notes`` are free lines printed
    under the rows for anything that is not a statistic. One write, as in ``banner``.
    """
    rows = [(str(label), str(value)) for label, value in rows]
    label_width = max((len(label) for label, _ in rows), default=0)
    value_width = max((len(value) for _, value in rows), default=0)
    rule = '-' * WIDTH
    lines = [f'  {label.ljust(label_width)}   {value.rjust(value_width)}' for label, value in rows]
    lines += [f'  {note}' for note in notes]
    print('\n'.join(['', rule, title, rule, *lines, rule]), flush=True)


def humanize_bytes(size) -> str:
    """``size`` in bytes at the largest unit that leaves it above 1, e.g. ``123.4 GB``."""
    size = float(size)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024 or unit == 'TB':
            return f'{size:,.0f} B' if unit == 'B' else f'{size:,.1f} {unit}'
        size /= 1024
