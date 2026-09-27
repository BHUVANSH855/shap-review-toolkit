from __future__ import annotations

import inspect
import random
import time

import numpy as np

PROTOCOLS = (
    "shape",
    "dtype",
    "strides",
    "getitem",
    "array",
    "len",
    "iter",
    "bool",
    "repr",
    "call",
)


class ProtocolObject:
    """Adversarial object whose protocol observations carry runtime provenance.

    ``phase`` is only descriptive metadata. A protocol becomes ``shap_observed``
    only when the actual Python call stack contains a SHAP frame while the
    protocol method/property is executing. This prevents a harness-side
    ``set_phase('shap')`` from fabricating target causality.
    """

    def __init__(
        self,
        rng: random.Random,
        protocol: str,
        *,
        mutation="none",
        reentry=False,
        exception=False,
        reentry_target=None,
        scenario="shape-dtype",
        execution_id=None,
        allow_harness_reentry=True,
    ):
        self.rng = rng
        self.protocol = protocol
        self.scenario = scenario
        self.mutation_mode = mutation
        self.reentry_mode = reentry
        self.exception_mode = exception
        self.reentry_target = reentry_target
        self.execution_id = execution_id
        self.allow_harness_reentry = allow_harness_reentry
        self.calls = 0
        self._protocol_event_records = []
        self.reentered = False
        self.mutated = False
        self.events = []
        self.mutation_events = []
        self.reentry_depth = 0
        self.state = {
            "shape_calls": 0,
            "reshape_calls": 0,
            "astype_calls": 0,
            "array_calls": 0,
            "getitem_calls": 0,
        }
        self.current_shape = (2,)
        self.current_dtype = "float64"
        self.current_values = np.array([0.0, 1.0], dtype=float)
        self.phase = "harness"
        self.active_shap_call_id = None
        self.active_shap_function = None
        self.active_shap_source = None

    def set_phase(self, phase: str):
        self.phase = phase

    def begin_shap_call(
        self,
        call_id: str,
        function: str = "TreeExplainer.shap_values",
        source: str | None = None,
    ):
        self.active_shap_call_id = call_id
        self.active_shap_function = function
        self.active_shap_source = source
        self.phase = "shap"

    def end_shap_call(self):
        self.active_shap_call_id = None
        self.active_shap_function = None
        self.active_shap_source = None

    def _runtime_provenance(self):
        frames = []
        shap_frames = []
        for frame in inspect.stack(context=0):
            module = frame.frame.f_globals.get("__name__", "")
            name = frame.function
            frames.append(f"{module}.{name}")
            if (
                module == "shap"
                or module.startswith("shap.")
                or "/shap/" in frame.filename.replace("\\", "/")
            ):
                shap_frames.append(f"{module}.{name}")
        return {
            "stack": frames[:40],
            "shap_frames": shap_frames[:20],
            "shap_frame_observed": bool(shap_frames),
            "nearest_shap_frame": shap_frames[0] if shap_frames else None,
        }

    def _touch(self, name):
        self.calls += 1
        source = self.phase
        prov = self._runtime_provenance()
        self.events.append(f"{name}:{source}")
        event_id = f"protocol-{self.calls}"
        causal = bool(prov["shap_frame_observed"] and self.active_shap_call_id)
        shap_frame = prov["shap_frames"][0] if prov["shap_frames"] else None
        event = {
            "event_id": event_id,
            "callback_event_id": event_id,
            "protocol": name,
            "phase": source,
            "execution_id": self.execution_id,
            "timestamp_ns": time.time_ns(),
            "shap_frame_observed": prov["shap_frame_observed"],
            "shap_call_id": self.active_shap_call_id,
            "active_shap_call_id": self.active_shap_call_id,
            "shap_function": self.active_shap_function,
            "shap_source": self.active_shap_source,
            "shap_frame": shap_frame,
            "nearest_shap_frame": prov.get("nearest_shap_frame"),
            "canonical_source": prov.get("nearest_shap_frame")
            or self.active_shap_source,
            "causal_to_active_shap_call": causal,
            "causal_confidence": "high" if causal else "none",
            "stack": prov["stack"],
            "shap_frames": prov["shap_frames"],
        }
        self._protocol_event_records.append(event)
        self.events.append(
            f"provenance:{name}:{'shap' if prov['shap_frame_observed'] else source}"
        )
        if self.exception_mode and name == self.protocol:
            raise RuntimeError(f"protocol-injected:{name}")
        if self.mutation_mode != "none":
            self._mutate(name, source, event)
        # Re-entry is permitted only from a real protocol callback occurring while
        # SHAP is on the runtime stack. Harness-side touches cannot trigger it.
        if (
            self.reentry_mode
            and name == self.protocol
            and self.reentry_target is not None
            and not self.reentered
            and (
                (prov["shap_frame_observed"] and self.active_shap_call_id is not None)
                or (self.allow_harness_reentry and source == "harness")
            )
        ):
            self.reentered = True
            self.reentry_depth += 1
            self.events.append(
                f"reentry:{name}:{'shap' if prov['shap_frame_observed'] else source}"
            )
            try:
                return self.reentry_target(self)
            finally:
                self.reentry_depth -= 1
        return None

    def _mutate(self, name, source, event):
        before_values = self.current_values.copy()
        before_shape = self.current_shape
        before_dtype = self.current_dtype
        before_object_id = id(self)
        before_payload_id = id(self.current_values)
        before_strides = tuple(self.current_values.strides)
        before_writeable = bool(self.current_values.flags.writeable)
        before_storage_shape = tuple(self.current_values.shape)
        before_storage_dtype = str(self.current_values.dtype)
        mode = self.mutation_mode
        if mode in {"on-access", "value"}:
            self.current_values[0] += 1.0
        elif mode == "shape":
            self.current_shape = (2, 1)
        elif mode == "dtype":
            self.current_dtype = (
                "float32" if self.current_dtype == "float64" else "float64"
            )
        elif mode == "strides":
            self.current_values = self.current_values[::-1]
        elif mode == "writeability":
            self.current_values.flags.writeable = False
        elif mode in {"identity", "buffer"}:
            self.current_values = np.array(self.current_values, copy=True)
        after_values = self.current_values.copy()
        after_shape = self.current_shape
        after_dtype = self.current_dtype
        after_payload_id = id(self.current_values)
        after_strides = tuple(self.current_values.strides)
        after_writeable = bool(self.current_values.flags.writeable)
        after_storage_shape = tuple(self.current_values.shape)
        after_storage_dtype = str(self.current_values.dtype)
        value_changed = not np.array_equal(before_values, after_values)
        metadata_changed = (
            before_shape != after_shape
            or before_dtype != after_dtype
            or before_writeable != after_writeable
        )
        payload_identity_changed = before_payload_id != after_payload_id
        buffer_changed = payload_identity_changed
        storage_shape_changed = before_storage_shape != after_storage_shape
        storage_dtype_changed = before_storage_dtype != after_storage_dtype
        changed = (
            value_changed
            or metadata_changed
            or payload_identity_changed
            or before_strides != after_strides
        )
        self.mutated = self.mutated or changed
        mutation_kind = {
            "identity": "payload_identity",
            "buffer": "buffer_replacement",
            "shape": "protocol_metadata",
            "dtype": "protocol_metadata",
        }.get(mode, "value" if mode in {"on-access", "value"} else mode)
        ev = {
            "type": mutation_kind,
            "protocol": name,
            "requested_mode": mode,
            "changed": changed,
            "value_before": before_values.tolist(),
            "value_after": after_values.tolist(),
            "value_changed": value_changed,
            "protocol_shape_before": list(before_shape),
            "protocol_shape_after": list(after_shape),
            "protocol_shape_changed": before_shape != after_shape,
            "protocol_dtype_before": before_dtype,
            "protocol_dtype_after": after_dtype,
            "protocol_dtype_changed": before_dtype != after_dtype,
            "storage_shape_before": list(before_storage_shape),
            "storage_shape_after": list(after_storage_shape),
            "storage_shape_changed": storage_shape_changed,
            "storage_dtype_before": before_storage_dtype,
            "storage_dtype_after": after_storage_dtype,
            "storage_dtype_changed": storage_dtype_changed,
            "strides_before": list(before_strides),
            "strides_after": list(after_strides),
            "strides_changed": before_strides != after_strides,
            "object_identity_before": before_object_id,
            "object_identity_after": id(self),
            "object_identity_changed": False,
            "payload_identity_before": before_payload_id,
            "payload_identity_after": after_payload_id,
            "payload_identity_changed": payload_identity_changed,
            "buffer_before": before_payload_id,
            "buffer_after": after_payload_id,
            "buffer_changed": buffer_changed,
            "writeable_before": before_writeable,
            "writeable_after": after_writeable,
            "writeable_changed": before_writeable != after_writeable,
            "mutation_semantics": "protocol-visible"
            if mode in {"shape", "dtype"}
            else "storage-backed"
            if mode
            in {"value", "on-access", "strides", "writeability", "identity", "buffer"}
            else "unknown",
            "phase": source,
            "execution_id": self.execution_id,
            "event_id": event["event_id"],
        }
        self.mutation_events.append(ev)
        self.events.append(f"mutation:{name}:{source}")

    @property
    def shape(self):
        self._touch("shape")
        return self.current_shape

    @property
    def ndim(self):
        self._touch("ndim")
        return len(self.current_shape)

    @property
    def size(self):
        self._touch("size")
        return int(np.prod(self.current_shape))

    def reshape(self, *shape):
        self.state["reshape_calls"] += 1
        self.events.append(f"reshape:{self.phase}")
        self.current_shape = tuple(shape)
        return self

    def astype(self, dtype, copy=True):
        self.state["astype_calls"] += 1
        self.events.append(f"astype:{self.phase}")
        if self.exception_mode and self.scenario == "array-coercion":
            raise RuntimeError("protocol-injected:astype")
        return np.zeros(self.current_shape or (2,), dtype=dtype)

    @property
    def dtype(self):
        self._touch("dtype")
        return self.current_dtype

    @property
    def strides(self):
        self._touch("strides")
        return tuple(self.current_values.strides)

    def __len__(self):
        self._touch("len")
        return 2

    def __getitem__(self, index):
        self.state["getitem_calls"] += 1
        self._touch("getitem")
        return 0.0

    def __array__(self, dtype=None):
        self.state["array_calls"] += 1
        self._touch("array")
        return np.asarray(self.current_values, dtype=dtype or float)

    def __iter__(self):
        self._touch("iter")
        return iter([0.0, 0.0])

    def __bool__(self):
        self._touch("bool")
        return True

    def __repr__(self):
        self._touch("repr")
        return "<ProtocolObject>"

    def __call__(self, *args, **kwargs):
        self._touch("call")
        return 0.0


def generate_protocol_case(rng: random.Random) -> dict:
    protocol = rng.choice(PROTOCOLS)
    mutation = rng.choice(
        [
            "none",
            "value",
            "shape",
            "dtype",
            "strides",
            "writeability",
            "identity",
            "buffer",
        ]
    )
    return {
        "protocol": protocol,
        "mutation": mutation,
        "reentry": rng.choice([False, True]),
        "exception": rng.choice([False, False, True]),
    }
