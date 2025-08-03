# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import math


def bind_events(component, events, fn, inputs=None, outputs=None):
    """
    Attach multiple Gradio event listeners to a component.

    Args:
        component: A Gradio component (e.g., gr.Textbox, gr.Slider).
        events (Iterable[str]): Event names to bind, e.g. ["submit", "blur"], ["change"], ["input", "change"].
        fn (callable): The function to call on each event.
        inputs (list|None): Inputs passed to `fn`.
        outputs (list|None): Outputs returned from `fn`.

    Returns:
        component: The same component, for optional chaining.
    """
    if not events:
        raise ValueError("`events` must be a non-empty iterable of event names.")

    # Bind each requested event if supported by the component
    for event in events:
        listener = getattr(component, event, None)
        if listener is None or not callable(listener):
            # Provide a clear error that names the missing event
            raise AttributeError(
                f"{component.__class__.__name__} does not support the '{event}' event."
            )
        listener(fn=fn, inputs=inputs, outputs=outputs)

    return component


def snap_to_base_value(v, base_v, min_v=None):
    """Snap value to nearest multiple of base_v, with optional minimum"""
    if base_v is None:
        base_v = 1   # avoid division by zero
    if v is None:
        v = 0  # handle None
    v = base_v * math.ceil(float(v) / base_v)
    return clamp_value(v, min_v=min_v)


def clamp_value(v, min_v=None, max_v=None, precision=None):
    """Clamp value to optional min/max"""
    if v is None:
        v = 0  # handle None
    if min_v is not None:
        v = max(v, min_v)
    if max_v is not None:
        v = min(v, max_v)
    if precision is not None:
        if precision == 0:
            v = int(v)
        else:
            print(f"precision: {precision} is not supported")
    return v
