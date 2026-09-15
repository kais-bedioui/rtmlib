"""TensorRT-backed RTMO, as an alternative to rtmlib's opencv/onnxruntime/
openvino backends.

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
from typing import Dict, List, Tuple

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
        self.output_names: List[str] = []
        self._allocators: Dict[str, _OutputAllocator] = {}
        self._input_ptr = 0
        self._input_nbytes = 0

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise ValueError(
                        'TRTEngine only supports single-input engines, '
                        f'found a second input: {name!r}')
                self.input_name = name
            else:
                self.output_names.append(name)
                allocator = _OutputAllocator()
                self.context.set_output_allocator(name, allocator)
                self._allocators[name] = allocator

        if self.input_name is None:
            raise ValueError(f'{engine_path} has no input tensor')

    def __call__(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the engine on `input_array` (already preprocessed: the
        exact dtype/shape the engine was built for) and return a dict of
        output tensor name -> numpy array."""
        input_array = np.ascontiguousarray(input_array)
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
            host_out = np.empty(shape, dtype=np.float32)
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
    """

    def __init__(self,
                onnx_model: str = None,
                engine_path: str = None,
                model_input_size: tuple = (640, 640),
                mean: tuple = None,
                std: tuple = None,
                nms_thr: float = 0.45,
                score_thr: float = 0.7,
                to_openpose: bool = False):
        """
        Args:
            onnx_model: Source .onnx path, used to build the engine if
                `engine_path` doesn't exist yet. Optional if `engine_path`
                already points at a prebuilt engine.
            engine_path: Path to a prebuilt (or to-be-built) .engine
                file. Defaults to `onnx_model` with its extension
                replaced by `.engine`.
            model_input_size, mean, std, nms_thr, score_thr, to_openpose:
                same as `RTMO` -- forwarded to the inherited
                `preprocess()`/`postprocess()`/`__call__()` unchanged.
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
            engine_path = os.path.splitext(onnx_model)[0] + '.engine'

        if not os.path.exists(engine_path):
            if onnx_model is None:
                raise FileNotFoundError(
                    f'{engine_path} does not exist, and no onnx_model was '
                    'given to build it.')
            build_engine(
                onnx_model,
                engine_path,
                input_shape=(1, 3, model_input_size[1], model_input_size[0]))

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
        return outputs
