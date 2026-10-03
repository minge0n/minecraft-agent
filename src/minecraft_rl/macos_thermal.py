"""Read the thermal state of an Apple Silicon Mac without root rights.

Two sources:

1. `thermal_pressure()`: the thermal pressure level of macOS, from the
   `com.apple.system.thermalpressurelevel` notification (0 nominal,
   1 moderate, 2 heavy, 3 trapping, 4 sleeping). macOS raises it only when
   it already limits performance.
2. `CpuTemperature`: the die temperature sensors of the SoC in degrees
   Celsius, through the HID event system of IOKit. These are the "PMU tdie"
   sensors that tools such as Stats and iStat Menus also read. The value
   rises long before the pressure level changes.

Both return None when the source is not available.
"""

import ctypes
import ctypes.util

THERMAL_NOTIFICATION = b"com.apple.system.thermalpressurelevel"
THERMAL_LEVELS = ("nominal", "moderate", "heavy", "trapping", "sleeping")

_UTF8 = 0x08000100
_NUMBER_SINT32 = 3
_APPLE_VENDOR_USAGE_PAGE = 0xFF00
_TEMPERATURE_SENSOR_USAGE = 5
_TEMPERATURE_EVENT = 15
_DIE_SENSOR_PREFIX = "PMU tdie"
_PLAUSIBLE_CELSIUS = (0.0, 150.0)


def thermal_pressure() -> int | None:
    """The current macOS thermal pressure level, or None."""
    library = ctypes.util.find_library("System")
    if library is None:
        return None
    system = ctypes.CDLL(library)
    token = ctypes.c_int()
    if system.notify_register_check(THERMAL_NOTIFICATION, ctypes.byref(token)):
        return None
    level = ctypes.c_uint64()
    status = system.notify_get_state(token, ctypes.byref(level))
    system.notify_cancel(token)
    return None if status else int(level.value)


class CpuTemperature:
    """The highest die temperature of the SoC. Create one instance and call
    `read()` as often as needed. The sensor list is found once."""

    def __init__(self) -> None:
        self._services: list[int] = []
        self._array = None
        try:
            self._load()
        except (OSError, AttributeError, ValueError):
            self._services = []

    def _load(self) -> None:
        iokit_path = ctypes.util.find_library("IOKit")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if iokit_path is None or cf_path is None:
            return
        iokit = ctypes.CDLL(iokit_path)
        cf = ctypes.CDLL(cf_path)
        pointer = ctypes.c_void_p
        cf.CFDictionaryCreateMutable.restype = pointer
        cf.CFDictionaryCreateMutable.argtypes = [
            pointer,
            ctypes.c_long,
            pointer,
            pointer,
        ]
        cf.CFNumberCreate.restype = pointer
        cf.CFNumberCreate.argtypes = [pointer, ctypes.c_int, pointer]
        cf.CFStringCreateWithCString.restype = pointer
        cf.CFStringCreateWithCString.argtypes = [
            pointer,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        cf.CFDictionarySetValue.argtypes = [pointer, pointer, pointer]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [pointer]
        cf.CFArrayGetValueAtIndex.restype = pointer
        cf.CFArrayGetValueAtIndex.argtypes = [pointer, ctypes.c_long]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [
            pointer,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        cf.CFRelease.argtypes = [pointer]
        iokit.IOHIDEventSystemClientCreate.restype = pointer
        iokit.IOHIDEventSystemClientCreate.argtypes = [pointer]
        iokit.IOHIDEventSystemClientSetMatching.argtypes = [pointer, pointer]
        iokit.IOHIDEventSystemClientCopyServices.restype = pointer
        iokit.IOHIDEventSystemClientCopyServices.argtypes = [pointer]
        iokit.IOHIDServiceClientCopyProperty.restype = pointer
        iokit.IOHIDServiceClientCopyProperty.argtypes = [pointer, pointer]
        iokit.IOHIDServiceClientCopyEvent.restype = pointer
        iokit.IOHIDServiceClientCopyEvent.argtypes = [
            pointer,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int64,
        ]
        iokit.IOHIDEventGetFloatValue.restype = ctypes.c_double
        iokit.IOHIDEventGetFloatValue.argtypes = [pointer, ctypes.c_int32]
        self._cf, self._iokit = cf, iokit

        def string(text: str) -> int:
            return cf.CFStringCreateWithCString(None, text.encode(), _UTF8)

        def number(value: int) -> int:
            raw = ctypes.c_int32(value)
            return cf.CFNumberCreate(None, _NUMBER_SINT32, ctypes.byref(raw))

        keys = pointer.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        values = pointer.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        matching = cf.CFDictionaryCreateMutable(
            None, 0, ctypes.addressof(keys), ctypes.addressof(values)
        )
        for key, value in (
            ("PrimaryUsagePage", _APPLE_VENDOR_USAGE_PAGE),
            ("PrimaryUsage", _TEMPERATURE_SENSOR_USAGE),
        ):
            key_ref, value_ref = string(key), number(value)
            cf.CFDictionarySetValue(matching, key_ref, value_ref)
            cf.CFRelease(key_ref)
            cf.CFRelease(value_ref)
        self._client = iokit.IOHIDEventSystemClientCreate(None)
        if not self._client:
            cf.CFRelease(matching)
            return
        iokit.IOHIDEventSystemClientSetMatching(self._client, matching)
        cf.CFRelease(matching)
        self._array = iokit.IOHIDEventSystemClientCopyServices(self._client)
        if not self._array:
            return
        product = string("Product")
        name = ctypes.create_string_buffer(256)
        for index in range(cf.CFArrayGetCount(self._array)):
            service = cf.CFArrayGetValueAtIndex(self._array, index)
            label = iokit.IOHIDServiceClientCopyProperty(service, product)
            if not label:
                continue
            found = cf.CFStringGetCString(label, name, len(name), _UTF8)
            cf.CFRelease(label)
            if found and name.value.decode().startswith(_DIE_SENSOR_PREFIX):
                self._services.append(service)
        cf.CFRelease(product)

    @property
    def available(self) -> bool:
        return bool(self._services)

    def read(self) -> float | None:
        """The highest plausible die temperature in degrees Celsius, or None."""
        highest = None
        low, high = _PLAUSIBLE_CELSIUS
        for service in self._services:
            event = self._iokit.IOHIDServiceClientCopyEvent(
                service, _TEMPERATURE_EVENT, 0, 0
            )
            if not event:
                continue
            value = self._iokit.IOHIDEventGetFloatValue(event, _TEMPERATURE_EVENT << 16)
            self._cf.CFRelease(event)
            if low < value < high and (highest is None or value > highest):
                highest = value
        return highest
