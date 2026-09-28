from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SHAPAxisSpec:
    """Canonical semantic axes for a SHAP tensor."""

    sample_axis: int = 0
    feature_axis: int = 1
    output_axis: int | None = None
    class_axis: int | None = None
    interaction_feature_axes: tuple[int, int] | None = None

    def normalize(self, ndim: int) -> SHAPAxisSpec:
        def n(axis: int) -> int:
            value = axis if axis >= 0 else ndim + axis
            if value < 0 or value >= ndim:
                raise ValueError(f"axis {axis} is invalid for rank {ndim}")
            return value

        sample = n(self.sample_axis)
        feature = n(self.feature_axis)
        output = None if self.output_axis is None else n(self.output_axis)
        pair = (
            None
            if self.interaction_feature_axes is None
            else tuple(n(a) for a in self.interaction_feature_axes)
        )
        class_axis = None if self.class_axis is None else n(self.class_axis)
        if sample == feature:
            raise ValueError("sample_axis and feature_axis must be distinct")
        if output is not None and output in {sample, feature}:
            raise ValueError(
                "output_axis must be distinct from sample_axis and feature_axis"
            )
        if pair is not None:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("interaction_feature_axes requires two distinct axes")
            if sample in pair:
                raise ValueError("interaction feature axes cannot include sample_axis")
            if output is not None and output in pair:
                raise ValueError("interaction feature axes cannot include output_axis")
        if class_axis is not None and (
            class_axis in {sample, feature}
            or (output is not None and class_axis != output)
        ):
            raise ValueError(
                "class_axis must be distinct from sample/feature and align with output_axis"
            )
        return SHAPAxisSpec(
            sample_axis=sample,
            feature_axis=feature,
            output_axis=output,
            class_axis=class_axis,
            interaction_feature_axes=pair,
        )


@dataclass(frozen=True)
class SHAPSemanticTensor:
    """One canonical semantic representation used by all SHAP comparison/oracle paths.

    ``values`` is the numeric SHAP contribution tensor. Axis semantics are carried
    alongside it so callers never need to infer meaning from equal dimensions.
    ``base_values`` and ``expected_value`` retain their original semantic forms.
    """

    values: np.ndarray
    axis_spec: SHAPAxisSpec
    base_values: np.ndarray | None = None
    expected_value: np.ndarray | None = None
    output_space: str = "raw"
    semantic_kind: str = "values"  # values | interaction
    source_api: str = "unknown"
    backend: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def interaction(self) -> bool:
        return self.semantic_kind == "interaction"

    @classmethod
    def from_values(
        cls,
        values: Any,
        *,
        base_values: Any = None,
        expected_value: Any = None,
        interaction: bool = False,
        axis_spec: SHAPAxisSpec | None = None,
        output_space: str = "raw",
        source_api: str = "unknown",
        backend: str | None = None,
        class_axis: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SHAPSemanticTensor:
        if (
            isinstance(values, (list, tuple))
            and values
            and all(hasattr(v, "shape") for v in values)
        ):
            arrays = [np.asarray(v, dtype=float) for v in values]
            if len({a.shape for a in arrays}) != 1:
                raise ValueError("legacy SHAP output list contains inconsistent shapes")
            arr = np.stack(arrays, axis=-1)
        else:
            arr = np.asarray(values, dtype=float)
        spec = (
            axis_spec
            or infer_axis_spec(arr, interaction=interaction, class_axis=class_axis)
        ).normalize(arr.ndim)
        return cls(
            arr,
            spec,
            None if base_values is None else np.asarray(base_values, dtype=float),
            None if expected_value is None else np.asarray(expected_value, dtype=float),
            output_space,
            "interaction" if interaction else "values",
            source_api,
            backend,
            metadata or {},
        )

    def reduce_contributions(self) -> np.ndarray:
        return reduce_contributions(
            self.values, interaction=self.interaction, axes=self.axis_spec
        )

    def reconstruction(self) -> np.ndarray:
        if self.base_values is None:
            raise ValueError("base_values are required for reconstruction")
        reconstructed = self.reduce_contributions()
        base = _align_base(self.base_values, reconstructed.shape, self.axis_spec)
        return reconstructed + base


def _align_base(
    base_values: np.ndarray, target_shape: tuple[int, ...], axes: SHAPAxisSpec
) -> np.ndarray:
    base = np.asarray(base_values, dtype=float)
    if base.shape == target_shape:
        return base
    if base.ndim == 0:
        return np.broadcast_to(base, target_shape)
    if base.ndim == 1:
        output_axis = (
            axes.output_axis
            if axes.output_axis is not None and axes.output_axis < len(target_shape)
            else (len(target_shape) - 1 if axes.output_axis is not None else None)
        )
        # Ambiguity guard: when n_samples == n_classes both the output_axis and
        # sample_axis branches would match, producing a silently wrong broadcast.
        # Raise an explicit error so the caller must supply an axis_spec.
        sample_matches = (
            axes.sample_axis < len(target_shape)
            and base.shape[0] == target_shape[axes.sample_axis]
        )
        output_matches = (
            output_axis is not None and base.shape[0] == target_shape[output_axis]
        )
        if output_matches and sample_matches and output_axis != axes.sample_axis:
            raise ValueError(
                f"Ambiguous 1-D baseline shape {base.shape}: matches both "
                f"output_axis={output_axis} and sample_axis={axes.sample_axis} "
                f"in target shape {target_shape}.  Provide an explicit axis_spec "
                f"so the correct semantic axis can be determined."
            )
        if output_matches:
            shape = [1] * len(target_shape)
            shape[output_axis] = base.shape[0]
            return np.broadcast_to(base.reshape(shape), target_shape)
        if base.shape[0] == 1:
            return np.broadcast_to(base.reshape([1] * len(target_shape)), target_shape)
        if sample_matches:
            shape = [1] * len(target_shape)
            shape[axes.sample_axis] = base.shape[0]
            return np.broadcast_to(base.reshape(shape), target_shape)
        raise ValueError(
            f"baseline vector shape {base.shape} is not semantically aligned to {target_shape}"
        )
    if base.ndim != len(target_shape):
        raise ValueError(
            f"baseline rank {base.ndim} cannot be promoted to {len(target_shape)} without explicit semantic axes"
        )
    for i, (b, t) in enumerate(zip(base.shape, target_shape)):
        if b not in (1, t):
            raise ValueError(f"baseline axis {i} shape {b} does not match target {t}")
    return np.broadcast_to(base, target_shape)


def semantic_align(
    left: np.ndarray,
    right: np.ndarray,
    *,
    axes: SHAPAxisSpec | None = None,
    role: str = "target",
):
    """Align semantic tensors using explicit SHAP axis contracts.

    This is the single alignment path used by additivity, output-space,
    expected-value and differential comparisons. Generic NumPy broadcasting is
    only permitted for scalar values or explicitly authorized singleton axes.
    Feature and interaction axes are never implicitly expanded.
    """
    a, b = np.asarray(left), np.asarray(right)
    if a.shape == b.shape:
        return (
            a,
            b,
            {
                "broadcast_applied": False,
                "source_shape": list(a.shape),
                "target_shape": list(b.shape),
                "semantic_axis": None,
                "axes": [],
                "reason": "exact-shape",
                "role": role,
            },
        )
    if a.ndim == 0 or b.ndim == 0:
        aa, bb = np.broadcast_arrays(a, b)
        return (
            aa,
            bb,
            {
                "broadcast_applied": True,
                "source_shape": list(a.shape),
                "target_shape": list(b.shape),
                "semantic_axis": "scalar",
                "axes": list(range(max(a.ndim, b.ndim))),
                "reason": "scalar semantic value",
                "role": role,
            },
        )
    if a.ndim != b.ndim:
        # A single-sample target is commonly represented as (outputs,) while the
        # reconstructed SHAP tensor is (1, outputs). This is the only non-scalar
        # rank reduction authorized here: the omitted axis must be a singleton
        # sample axis. No arbitrary rank promotion is allowed.
        if (
            a.ndim == b.ndim + 1
            and a.shape[0] == 1
            and tuple(a.shape[1:]) == tuple(b.shape)
        ):
            bb = b.reshape((1,) + b.shape)
            return (
                a,
                bb,
                {
                    "broadcast_applied": False,
                    "source_shape": list(a.shape),
                    "target_shape": list(b.shape),
                    "semantic_axis": "sample_singleton_omission",
                    "axes": [0],
                    "reason": "single-sample target omitted singleton sample axis",
                    "role": role,
                },
            )
        if (
            b.ndim == a.ndim + 1
            and b.shape[0] == 1
            and tuple(b.shape[1:]) == tuple(a.shape)
        ):
            aa = a.reshape((1,) + a.shape)
            return (
                aa,
                b,
                {
                    "broadcast_applied": False,
                    "source_shape": list(a.shape),
                    "target_shape": list(b.shape),
                    "semantic_axis": "sample_singleton_omission",
                    "axes": [0],
                    "reason": "single-sample source omitted singleton sample axis",
                    "role": role,
                },
            )
        raise ValueError(
            f"rank mismatch is not contract-authorized: {a.shape} vs {b.shape}"
        )

    if axes is None and b.ndim == 1:
        # One-dimensional baseline/target vectors have no feature-axis contract.
        spec = None
        protected = set()
    else:
        spec = (axes or infer_axis_spec(b)).normalize(b.ndim)
        protected = {spec.feature_axis}
        if spec.interaction_feature_axes:
            protected.update(spec.interaction_feature_axes)
    differing = []
    for i, (x, y) in enumerate(zip(a.shape, b.shape)):
        if x == y:
            continue
        differing.append(i)
        if i in protected:
            raise ValueError(
                f"feature/interaction axis {i} cannot be implicitly broadcast: {a.shape} vs {b.shape}"
            )
        if x != 1 and y != 1:
            raise ValueError(
                f"non-singleton semantic axis {i} is incompatible: {a.shape} vs {b.shape}"
            )

    aa, bb = np.broadcast_arrays(a, b)
    semantic_axes = []
    for axis in differing:
        label = (
            "vector"
            if spec is None
            else (
                "sample"
                if axis == spec.sample_axis
                else (
                    "output"
                    if axis == spec.output_axis
                    else (
                        "class" if axis == spec.class_axis else "non-feature-semantic"
                    )
                )
            )
        )
        semantic_axes.append({"axis": axis, "role": label})
    return (
        aa,
        bb,
        {
            "broadcast_applied": True,
            "source_shape": list(a.shape),
            "target_shape": list(b.shape),
            "semantic_axis": "singleton_dimension",
            "axes": differing,
            "semantic_axes": semantic_axes,
            "reason": "singleton expansion on non-feature semantic axes",
            "role": role,
        },
    )


def infer_axis_spec(
    values: np.ndarray, *, interaction: bool = False, class_axis: int | None = None
) -> SHAPAxisSpec:
    ndim = values.ndim
    if ndim == 2:
        return SHAPAxisSpec()
    if interaction:
        if ndim == 3:
            return SHAPAxisSpec(interaction_feature_axes=(1, 2))
        if ndim == 4:
            return SHAPAxisSpec(output_axis=3, interaction_feature_axes=(1, 2))
        raise ValueError(
            f"unsupported canonical interaction tensor rank: {ndim}; provide axis_spec"
        )
    if ndim == 3:
        return SHAPAxisSpec(
            output_axis=2, class_axis=class_axis if class_axis is not None else 2
        )
    raise ValueError(
        f"unsupported canonical SHAP tensor rank: {ndim}; provide axis_spec"
    )


def reduce_contributions(
    values: np.ndarray, *, interaction: bool = False, axes: SHAPAxisSpec | None = None
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    spec = (axes or infer_axis_spec(arr, interaction=interaction)).normalize(arr.ndim)
    if interaction:
        pair = spec.interaction_feature_axes
        if pair is None or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(
                "interaction tensor requires two distinct interaction feature axes"
            )
        if arr.shape[pair[0]] != arr.shape[pair[1]]:
            raise ValueError("interaction feature axes must have equal sizes")
        return arr.sum(axis=tuple(sorted(pair)))
    return arr.sum(axis=spec.feature_axis)
