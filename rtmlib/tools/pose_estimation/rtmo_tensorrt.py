"""TensorRT-backed RTMO, as an alternative to rtmlib's opencv/onnxruntime/
openvino backends.

**Deployment status**: verified end-to-end on x86 + an NVIDIA dGPU (RTX
500 Ada Generation Laptop GPU) -- correct output (person count and
keypoints match the onnxruntime backend), 74.5 fps full pipeline on
`rtmo-m`, beating onnxruntime's CUDA EP (48.9 fps) for the same tier.
Not yet wired into rtmlib's `backend=` dispatch (`tools/base.py`) or the
benchmark harness in the sibling `worktree-rtmlib-benchmark` -- import
`RTMOTensorRT` directly, it isn't reachable via `backend='tensorrt'` on
the existing classes.

**NOT yet verified on Jetson**, despite Jetson TensorRT being discussed
elsewhere in this project (see `worktree-rtmlib-benchmark`'s
`BENCHMARK_REPORT.md` §11.1): that work validated raw TensorRT via the
`trtexec` CLI directly on a Jetson AGX Orin (JetPack 7.2, TensorRT
10.16.2) -- a different code path from this module, which drives the
TensorRT Python API (`Builder.create_network()`, `IOutputAllocator`,
`execute_async_v3`) and the `cuda-python` package (`cuda.bindings.
runtime`) directly. Neither of those has been exercised on a Jetson
through *this* class. Known open questions before trusting it there:
whether `cuda-python`'s Tegra/Jetson wheel matches this API shape, and
whether TensorRT 10.16.2's Python builder API behaves identically to
whatever version this was built against on x86 (the FP16-flag removal
is already handled defensively -- see `build_engine()` -- but that's
the one version difference that was actually hit and fixed, not a
guarantee every other call in this file is equally version-safe).
Build and run `build_engine()` + `RTMOTensorRT` directly on a Jetson
before relying on this there.

TensorRT engines are hardware- and TensorRT-version-specific and must be
built ahead of time on the machine they'll run on (an engine built on one
GPU/TensorRT version will not load on another) -- unlike rtmlib's other
backends, which compile an onnxruntime/openvino session from a portable
.onnx file at construction time. `build_engine()` does that ahead-of-time
build; `RTMOTensorRT` then just loads and runs the resulting .engine file.

Requires the `tensorrt` and `cuda-python` packages (not rtmlib
dependencies -- install separately: `pip install tensorrt cuda-python`).

RTMO's ONNX graph has a genuinely data-dependent number of candidate
detections (its output shape isn't fixed even for a fixed input shape,
confirmed by inspecting the parsed network: `dets` is `(1, -1, 5)` and
`keypoints` is `(1, -1, 17, 3)`, still dynamic after pinning the input
shape). TensorRT's `IOutputAllocator` callback mechanism -- not a
pre-sized buffer -- is required for these two outputs; a static
`set_tensor_address` buffer (fine for the input, and for most other
rtmlib models) silently reads garbage or crashes for RTMO's outputs.
"""
import os
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "RTMOTensorRT requires the 'tensorrt' package: pip install "
        'tensorrt') from e

try:
    from cuda.bindings import runtime as cudart
except ImportError as e:
    raise ImportError("RTMOTensorRT requires the 'cuda-python' package: "
                      'pip install cuda-python') from e

from ..file import download_checkpoint
from .rtmo import RTMO


def _cuda_check(ret):
    """cuda-python calls return either just an error code, or (error,
    *values) -- normalize both into "the value(s), raising on error"."""
    err, rest = (ret[0], ret[1:]) if isinstance(ret, tuple) else (ret, ())
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f'CUDA error: {err}')
    if len(rest) == 1:
        return rest[0]
    return rest


def build_engine(onnx_path: str,
                 engine_path: str,
                 input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                 workspace_mb: int = 2048,
                 fp16: bool = False,
                 force: bool = False) -> str:
    """Build (or reuse a cached) TensorRT engine from an ONNX model with a
    fixed input shape, using TensorRT's Python builder API directly --
    no `trtexec` CLI dependency, so this works anywhere the `tensorrt`
    pip package is installed.

    RTMO's ONNX export declares a dynamic batch dimension; this pins it
    to `input_shape` via an optimization profile (min == opt == max) --
    the same fix rtmlib's OpenVINO NPU backend needs, for the same
    reason (see `rtmlib/tools/base.py`).

    Args:
        onnx_path: Path to the source .onnx file (e.g. from rtmlib's
            model cache).
        engine_path: Where to write the serialized engine. If it already
            exists, it's reused as-is (pass `force=True` to rebuild).
        input_shape: Fixed (N, C, H, W) input shape to build for.
        workspace_mb: TensorRT builder workspace size limit, in MiB.
        fp16: Request the classic `BuilderFlag.FP16` precision. Silently
            (with a `warnings.warn`) downgraded to FP32 on TensorRT
            builds that don't expose that flag -- **TensorRT 11 dropped
            it** in favor of "strongly typed" networks that match the
            ONNX's own tensor dtypes; Roboflow's `rfdetr` package hits
            this exact issue and falls back the same way (see
            `rfdetr.export._tensorrt.build_engine` for the same probe-
            and-fall-back pattern independently arrived at there). On a
            TensorRT build old enough to still have the flag (e.g.
            TensorRT 10.x, as JetPack 7.2 ships), this Just Works and is
            a real, no-caveats speedup -- confirmed via `trtexec` on a
            Jetson AGX Orin: RTMO-m went from 70.2 to 123.0 qps.
            See `build_engine_strongly_typed_fp16` for the TensorRT-11+
            alternative -- it's real (builds a genuine mixed-precision
            engine, correct output), but empirically gave *zero*
            measured speedup on an Ada Lovelace GPU + TensorRT 11.3
            (RTMO's Conv layers -- most of its FLOPs -- have no
            low-precision kernel for a strongly typed network there, so
            they end up staying FP32 anyway). It's opt-in, not the
            `fp16=True` default here, because of that: verify it
            actually helps on your own hardware before relying on it.
        force: Rebuild even if `engine_path` already exists.

    Returns:
        `engine_path`.
    """
    if os.path.exists(engine_path) and not force:
        return engine_path

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = '\n'.join(
                str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'Failed to parse {onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * 1024 * 1024)

    if fp16:
        if hasattr(trt.BuilderFlag, 'FP16'):
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            warnings.warn(
                f'TensorRT {getattr(trt, "__version__", "?")} has no '
                "BuilderFlag.FP16 (removed in TensorRT 11+ -- see "
                'build_engine()\'s fp16 docstring). Building FP32 '
                'instead. Pass fp16=False to silence this warning, or '
                'see build_engine_strongly_typed_fp16 for the (opt-in, '
                'not always faster) TensorRT-11+ alternative.',
                stacklevel=2)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT engine build failed for {onnx_path}')

    out_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    return engine_path


def build_engine_strongly_typed_fp16(
    onnx_path: str,
    engine_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    workspace_mb: int = 2048,
    op_block_list: Sequence[str] = ('Conv', 'ConvTranspose',
                                    'NonMaxSuppression'),
    force: bool = False,
) -> str:
    """Opt-in alternative to `build_engine(..., fp16=True)` for TensorRT
    builds without `BuilderFlag.FP16` (TensorRT 11+, see that function's
    docstring) -- builds a genuine mixed FP16/FP32 engine by casting the
    ONNX to FP16 first (keeping `op_block_list` in FP32) and compiling it
    as a "strongly typed" network, which respects the ONNX's own tensor
    dtypes instead of a global precision flag.

    This *works* (produces a correct, running engine -- verified on
    RTMO-m: same person count and near-identical keypoints as the FP32
    engine, off by sub-pixel amounts) but is not necessarily *faster*:
    on an Ada Lovelace GPU + TensorRT 11.3, it measured within noise of
    the plain FP32 engine (73.4 vs 70.2 qps) because TensorRT reported
    "No low-precision conv kernel available for this strongly-typed
    Conv/ConvTranspose" and forced RTMO's Conv layers -- the majority of
    its FLOPs -- back to FP32 regardless, which is exactly why `Conv`
    and `ConvTranspose` are in the default `op_block_list`: leaving them
    out doesn't buy real FP16 Conv execution here, just build failures.
    `NonMaxSuppression` is blocked because its `IoUThreshold` input is
    required by the ONNX op spec to stay float32 -- RTMO's graph
    contains a real NMS node internally, not just Python-side NMS.

    Requires `onnx` and `onnxconverter-common` (not rtmlib or
    RTMOTensorRT dependencies -- install separately if you want to try
    this path: `pip install onnx onnxconverter-common`).

    Whether this is worth using depends entirely on your own GPU's
    kernel support for low-precision strongly-typed Conv -- benchmark it
    against `build_engine(..., fp16=False)` before relying on it.
    """
    if os.path.exists(engine_path) and not force:
        return engine_path

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        raise ImportError(
            'build_engine_strongly_typed_fp16 requires the onnx and '
            'onnxconverter-common packages: pip install onnx '
            'onnxconverter-common') from e

    fp16_onnx_path = os.path.splitext(engine_path)[0] + '_fp16_cast.onnx'
    model_fp16 = float16.convert_float_to_float16(
        onnx.load(onnx_path), keep_io_types=False,
        op_block_list=list(op_block_list))
    onnx.save(model_fp16, fp16_onnx_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)

    with open(fp16_onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = '\n'.join(
                str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(
                f'Failed to parse {fp16_onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * 1024 * 1024)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            f'TensorRT strongly-typed FP16 engine build failed for '
            f'{fp16_onnx_path}. This model/op combination may need a '
            'different op_block_list -- check the build log above for '
            'which node TensorRT rejected.')

    out_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    return engine_path


class _OutputAllocator(trt.IOutputAllocator):
    """Lazily grows a device buffer to whatever size TensorRT asks for at
    execution time, and records the output's actual shape once known --
    required for RTMO's data-dependent output tensors (see module
    docstring); a fixed pre-sized buffer isn't valid for these."""

    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.ptr = 0
        self.capacity = 0
        self.shape = None

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if size > self.capacity:
            if self.ptr:
                cudart.cudaFree(self.ptr)
            self.ptr = int(_cuda_check(cudart.cudaMalloc(size)))
            self.capacity = size
        return self.ptr

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(shape)

    def __del__(self):
        if self.ptr:
            cudart.cudaFree(self.ptr)


class TRTEngine:
    """Runs a single-input TensorRT engine with cuda-python for device
    memory management -- no PyTorch or pycuda dependency. Output tensors
    are read via `IOutputAllocator` so this handles both static and
    data-dependent output shapes without needing to know which in
    advance."""

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'Failed to load TensorRT engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.stream = _cuda_check(cudart.cudaStreamCreate())

        self.input_name = None
        self.input_dtype = None
        self.output_names: List[str] = []
        self._output_dtypes: Dict[str, np.dtype] = {}
        self._allocators: Dict[str, _OutputAllocator] = {}
        self._input_ptr = 0
        self._input_nbytes = 0

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise ValueError(
                        'TRTEngine only supports single-input engines, '
                        f'found a second input: {name!r}')
                self.input_name = name
                self.input_dtype = dtype
            else:
                self.output_names.append(name)
                self._output_dtypes[name] = dtype
                allocator = _OutputAllocator()
                self.context.set_output_allocator(name, allocator)
                self._allocators[name] = allocator

        if self.input_name is None:
            raise ValueError(f'{engine_path} has no input tensor')

    def __call__(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the engine on `input_array` (already preprocessed to the
        engine's expected input shape -- dtype is cast automatically to
        whatever the engine was built for, e.g. float16 for an FP16
        engine) and return a dict of output tensor name -> numpy array,
        each in that output's own dtype (not necessarily float32 --
        callers of `RTMOTensorRT` don't need to care, but a raw
        `TRTEngine` caller does)."""
        input_array = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        self.context.set_input_shape(self.input_name, input_array.shape)

        nbytes = input_array.nbytes
        if nbytes > self._input_nbytes:
            if self._input_ptr:
                cudart.cudaFree(self._input_ptr)
            self._input_ptr = int(_cuda_check(cudart.cudaMalloc(nbytes)))
            self._input_nbytes = nbytes
        self.context.set_tensor_address(self.input_name, self._input_ptr)

        _cuda_check(
            cudart.cudaMemcpyAsync(
                self._input_ptr, input_array.ctypes.data, nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))

        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError('TensorRT execute_async_v3 failed')
        _cuda_check(cudart.cudaStreamSynchronize(self.stream))

        outputs = {}
        for name in self.output_names:
            allocator = self._allocators[name]
            shape = allocator.shape
            host_out = np.empty(shape, dtype=self._output_dtypes[name])
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    host_out.ctypes.data, allocator.ptr, host_out.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream))
            outputs[name] = host_out
        _cuda_check(cudart.cudaStreamSynchronize(self.stream))

        return outputs

    def __del__(self):
        if getattr(self, '_input_ptr', 0):
            cudart.cudaFree(self._input_ptr)
        stream = getattr(self, 'stream', None)
        if stream:
            cudart.cudaStreamDestroy(stream)


class RTMOTensorRT(RTMO):
    """RTMO pose estimator backed by a prebuilt TensorRT engine.

    Reuses `RTMO.preprocess()`/`postprocess()` unchanged (pure numpy/
    opencv, backend-agnostic) -- only `inference()`, the one method
    rtmlib's `BaseTool` dispatches per backend, is overridden. Does not
    go through `BaseTool.__init__`'s onnxruntime/openvino/opencv session
    setup at all, since a TensorRT engine is loaded, not compiled from
    an .onnx file at construction time.

    Example:
        from rtmlib.tools.pose_estimation.rtmo_tensorrt import (
            RTMOTensorRT, build_engine)

        engine = build_engine('rtmo-m.onnx', 'rtmo-m.engine')
        pose = RTMOTensorRT(engine_path=engine)
        keypoints, scores = pose(img)

        # FP16: real speedup on TensorRT builds old enough to still have
        # BuilderFlag.FP16 (e.g. TensorRT 10.x); a no-op fallback to FP32
        # (with a warning) on TensorRT 11+, which dropped it -- see
        # build_engine()'s fp16 docstring for why, and
        # build_engine_strongly_typed_fp16 for the (not always faster,
        # verify on your own hardware) TensorRT-11+ alternative.
        pose = RTMOTensorRT(onnx_model='rtmo-m.onnx', fp16=True)
    """

    def __init__(self,
                onnx_model: str = None,
                engine_path: str = None,
                model_input_size: tuple = (640, 640),
                mean: tuple = None,
                std: tuple = None,
                nms_thr: float = 0.45,
                score_thr: float = 0.7,
                to_openpose: bool = False,
                fp16: bool = False):
        """
        Args:
            onnx_model: Source .onnx path, used to build the engine if
                `engine_path` doesn't exist yet. Optional if `engine_path`
                already points at a prebuilt engine.
            engine_path: Path to a prebuilt (or to-be-built) .engine
                file. Defaults to `onnx_model` with its extension
                replaced by `.engine` (`_fp16.engine` if `fp16=True`, so
                the two precisions don't collide on the same cache path).
            model_input_size, mean, std, nms_thr, score_thr, to_openpose:
                same as `RTMO` -- forwarded to the inherited
                `preprocess()`/`postprocess()`/`__call__()` unchanged.
            fp16: Passed to `build_engine()` -- see its docstring. Only
                takes effect while building a new engine; has no effect
                if `engine_path` already exists (that engine's actual
                precision is whatever it was built with).
        """
        if onnx_model is not None and not os.path.exists(onnx_model):
            # Same convention as BaseTool.__init__: onnx_model may be a
            # download URL (as in the model zoo / *_demo.py scripts), in
            # which case it's fetched once and cached locally.
            onnx_model = download_checkpoint(onnx_model)

        if engine_path is None:
            if onnx_model is None:
                raise ValueError(
                    'Provide either onnx_model (to build an engine) or '
                    'engine_path (a prebuilt one).')
            suffix = '_fp16.engine' if fp16 else '.engine'
            engine_path = os.path.splitext(onnx_model)[0] + suffix

        if not os.path.exists(engine_path):
            if onnx_model is None:
                raise FileNotFoundError(
                    f'{engine_path} does not exist, and no onnx_model was '
                    'given to build it.')
            build_engine(
                onnx_model,
                engine_path,
                input_shape=(1, 3, model_input_size[1], model_input_size[0]),
                fp16=fp16)

        self.engine_runner = TRTEngine(engine_path)

        # Mirror BaseTool's bookkeeping without going through its
        # opencv/onnxruntime/openvino session-construction branches.
        self.onnx_model = onnx_model
        self.engine_path = engine_path
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = 'tensorrt'
        self.device = 'cuda'

        self.to_openpose = to_openpose
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def inference(self, img: np.ndarray):
        # Same CHW / float32 / batch-of-1 conversion BaseTool.inference()
        # does for the onnxruntime/openvino backends this replaces.
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input_array = img[None, :, :, :]

        raw_outputs = self.engine_runner(input_array)

        # RTMO's ONNX graph has exactly 2 outputs: a (1, N, 5) box+score
        # tensor and a (1, N, 17, 3) keypoint tensor. Sort by rank
        # instead of hardcoding names, so this doesn't depend on the
        # exporter's exact naming (rtmo.py's own postprocess() expects
        # them in this [det_outputs, pose_outputs] order).
        outputs = sorted(raw_outputs.values(), key=lambda a: a.ndim)
        # Cast back to float32 regardless of the engine's internal
        # precision (e.g. a mixed FP16/FP32 engine, see build_engine's
        # `fp16` option) -- RTMO.postprocess() and everything downstream
        # of it (multiclass_nms, draw_skeleton, ...) expects the same
        # float32 arrays the onnxruntime/openvino backends produce.
        return [o.astype(np.float32, copy=False) for o in outputs]
