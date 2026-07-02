import base64
import os
from collections import OrderedDict
from io import BytesIO
from threading import Lock
from typing import Optional, Tuple, Union

from PIL import Image

ImageInput = Union[Image.Image, str]

# Process-wide LRU for path-keyed images only. Paths are content-addressed via
# (abspath, mtime_ns, size), which stays valid across calls. PIL objects are
# deliberately NOT cached: object identity (id()) is reusable after GC, so an
# id-keyed entry can silently return another image's bytes.


def _cache_max_items() -> int:
    value = os.getenv("LMMS_IMAGE_PATH_CACHE_MAX_ITEMS", "256")
    try:
        return max(0, int(value))
    except ValueError:
        return 256


_PATH_CACHE_MAX_ITEMS = _cache_max_items()
_PATH_BASE64_CACHE: "OrderedDict[Tuple[object, ...], str]" = OrderedDict()
_PATH_CACHE_LOCK = Lock()


def _normalize_format(image_format: str) -> str:
    return image_format.upper()


def _build_path_cache_key(
    path: str,
    *,
    image_format: str,
    convert_rgb: bool,
    quality: Optional[int],
) -> Tuple[object, ...]:
    abs_path = os.path.abspath(path)
    try:
        stat = os.stat(abs_path)
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
    except OSError:
        mtime_ns = None
        size = None
    return (abs_path, mtime_ns, size, image_format, convert_rgb, quality)


def _lookup_path_cache(key: Tuple[object, ...]) -> Optional[str]:
    with _PATH_CACHE_LOCK:
        cached = _PATH_BASE64_CACHE.get(key)
        if cached is not None:
            _PATH_BASE64_CACHE.move_to_end(key)
        return cached


def _store_path_cache(key: Tuple[object, ...], value: str) -> None:
    if _PATH_CACHE_MAX_ITEMS <= 0:
        return
    with _PATH_CACHE_LOCK:
        _PATH_BASE64_CACHE[key] = value
        _PATH_BASE64_CACHE.move_to_end(key)
        while len(_PATH_BASE64_CACHE) > _PATH_CACHE_MAX_ITEMS:
            _PATH_BASE64_CACHE.popitem(last=False)


def _encode_pil_image_to_bytes(image: Image.Image, *, image_format: str, quality: Optional[int]) -> bytes:
    output_buffer = BytesIO()
    save_kwargs = {}
    if quality is not None and image_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
    image.save(output_buffer, format=image_format, **save_kwargs)
    return output_buffer.getvalue()


def encode_image_to_bytes(
    image: ImageInput,
    *,
    image_format: str = "PNG",
    convert_rgb: bool = False,
    quality: Optional[int] = None,
    copy_if_pil: bool = False,
) -> bytes:
    normalized_format = _normalize_format(image_format)

    if isinstance(image, str):
        with Image.open(image) as loaded_image:
            working_image = loaded_image.convert("RGB") if convert_rgb else loaded_image
            return _encode_pil_image_to_bytes(working_image, image_format=normalized_format, quality=quality)

    working_image = image.copy() if copy_if_pil else image
    if convert_rgb and working_image.mode != "RGB":
        working_image = working_image.convert("RGB")
    return _encode_pil_image_to_bytes(working_image, image_format=normalized_format, quality=quality)


def encode_image_to_base64(
    image: ImageInput,
    *,
    image_format: str = "PNG",
    convert_rgb: bool = False,
    quality: Optional[int] = None,
    copy_if_pil: bool = False,
    use_path_cache: bool = True,
) -> str:
    normalized_format = _normalize_format(image_format)

    cache_key: Optional[Tuple[object, ...]] = None
    if use_path_cache and isinstance(image, str):
        cache_key = _build_path_cache_key(
            image,
            image_format=normalized_format,
            convert_rgb=convert_rgb,
            quality=quality,
        )
        cached = _lookup_path_cache(cache_key)
        if cached is not None:
            return cached

    encoded_bytes = encode_image_to_bytes(
        image,
        image_format=normalized_format,
        convert_rgb=convert_rgb,
        quality=quality,
        copy_if_pil=copy_if_pil,
    )
    base64_str = base64.b64encode(encoded_bytes).decode("utf-8")

    if cache_key is not None:
        _store_path_cache(cache_key, base64_str)

    return base64_str


def encode_image_to_data_url(
    image: ImageInput,
    *,
    image_format: str = "PNG",
    mime_type: Optional[str] = None,
    convert_rgb: bool = False,
    quality: Optional[int] = None,
    copy_if_pil: bool = False,
    use_path_cache: bool = True,
) -> str:
    base64_str = encode_image_to_base64(
        image,
        image_format=image_format,
        convert_rgb=convert_rgb,
        quality=quality,
        copy_if_pil=copy_if_pil,
        use_path_cache=use_path_cache,
    )
    if mime_type is None:
        normalized = image_format.lower()
        if normalized == "jpg":
            normalized = "jpeg"
        mime_type = f"image/{normalized}"
    return f"data:{mime_type};base64,{base64_str}"


def encode_image_to_base64_with_size_limit(
    image: ImageInput,
    *,
    max_size_bytes: int,
    image_format: str = "PNG",
    convert_rgb: bool = True,
    quality: Optional[int] = None,
    copy_if_pil: bool = True,
    resize_factor: float = 0.75,
    min_side: int = 100,
    resample=Image.Resampling.LANCZOS,
) -> str:
    normalized_format = _normalize_format(image_format)

    if isinstance(image, str):
        with Image.open(image) as loaded_image:
            working_image = loaded_image.convert("RGB") if convert_rgb else loaded_image.copy()
    else:
        working_image = image.copy() if copy_if_pil else image
        if convert_rgb and working_image.mode != "RGB":
            working_image = working_image.convert("RGB")

    encoded_bytes = _encode_pil_image_to_bytes(working_image, image_format=normalized_format, quality=quality)
    while len(encoded_bytes) > max_size_bytes and working_image.size[0] > min_side and working_image.size[1] > min_side:
        new_size = (int(working_image.size[0] * resize_factor), int(working_image.size[1] * resize_factor))
        working_image = working_image.resize(new_size, resample)
        encoded_bytes = _encode_pil_image_to_bytes(working_image, image_format=normalized_format, quality=quality)

    return base64.b64encode(encoded_bytes).decode("utf-8")
