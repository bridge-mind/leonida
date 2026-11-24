from __future__ import annotations

"""Synthetic unit tests for tools/vk_subgroup_info.py.

Standard library only; no GPU, no Vulkan loader DLL, and no game data: the Vulkan loader ctypes API is replaced
with an in-memory fake that validates sType/pNext chains and writes real bytes into the caller's ctypes buffers.

    python -I tests/test_vk_subgroup_info.py
"""
import contextlib
import ctypes
from dataclasses import dataclass
import io
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import vk_subgroup_info as vk_info  # noqa: E402


def _ptr(arg) -> int | None:
    """Resolve a ctypes pointer, array, byref() argument, or None to its integer address."""
    return ctypes.cast(arg, ctypes.c_void_p).value


class _VkProc:
    """Callable stand-in for a ctypes CDLL symbol supporting argtypes assignment and arity checks."""

    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None

    def __call__(self, *args):
        if self.argtypes is not None and len(args) != len(self.argtypes):
            raise AssertionError(f"expected {len(self.argtypes)} arguments, got {len(args)}")
        return self._fn(*args)


@dataclass
class FakeDevice:
    """Synthetic physical device configuration exposed through FakeVulkan."""

    name: str | bytes = "Synthetic GPU"
    api_version: int = vk_info.VK_API_VERSION_1_3
    extension_names: tuple[bytes, ...] = ()
    device_type: int = vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    dynamic_rendering: bool = True
    synchronization2: bool = True
    timeline_semaphore: bool = True
    separate_depth_stencil_layouts: bool = True
    buffer_device_address: bool = True
    subgroup_size_control: bool = True
    queue_flags: tuple[int, ...] = (vk_info.VK_QUEUE_GRAPHICS_BIT,)
    subgroup_size: int = 32
    subgroup_supported_stages: int = (
        vk_info.VK_SHADER_STAGE_COMPUTE_BIT | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT
    )
    min_subgroup_size: int = 32
    max_subgroup_size: int = 64
    max_compute_workgroup_subgroups: int = 1024
    required_subgroup_size_stages: int = (
        vk_info.VK_SHADER_STAGE_COMPUTE_BIT | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT
    )
    fragment_shader_barycentric: bool = True
    max_push_constants_size: int = 256
    driver_id: int = 1      # VK_DRIVER_ID_AMD_PROPRIETARY
    vendor_id: int = 0
    device_id: int = 0


class FakeVulkan:
    """In-memory fake Vulkan loader writing real bytes into ctypes buffers and checking pNext chains."""

    INSTANCE_HANDLE = 0x10000001

    def __init__(self, devices=(), create_instance_result=0, enumerate_hook=None):
        self._devices = {}
        self._device_handles = []
        for i, dev in enumerate(devices):
            handle = 0x20000000 + i + 1
            self._devices[handle] = dev
            self._device_handles.append(handle)
        self.create_instance_result = create_instance_result
        self.enumerate_hook = enumerate_hook
        self.destroyed_instances = []
        self.enum_calls = 0
        self.props_calls = []
        self.features_calls = []
        self.queue_calls = []
        self.vkCreateInstance = _VkProc(self._create_instance)
        self.vkEnumeratePhysicalDevices = _VkProc(self._enumerate_physical_devices)
        self.vkEnumerateDeviceExtensionProperties = _VkProc(self._enumerate_device_extension_properties)
        self.vkGetPhysicalDeviceProperties2 = _VkProc(self._get_physical_device_properties2)
        self.vkGetPhysicalDeviceFeatures2 = _VkProc(self._get_physical_device_features2)
        self.vkGetPhysicalDeviceQueueFamilyProperties = _VkProc(self._get_queue_family_properties)
        self.vkDestroyInstance = _VkProc(self._destroy_instance)

    @staticmethod
    def _iter_chain(head_addr: int):
        addr = head_addr
        seen = set()
        while addr:
            if addr in seen:
                raise AssertionError("cycle in pNext chain")
            seen.add(addr)
            stype = ctypes.c_uint32.from_address(addr).value
            yield stype, addr
            addr = ctypes.c_void_p.from_address(addr + 8).value

    def _create_instance(self, p_info, _p_alloc, p_inst):
        info_addr = _ptr(p_info)
        if not info_addr:
            raise AssertionError("vkCreateInstance called with null pCreateInfo")
        info = vk_info.InstanceInfo.from_address(info_addr)
        if info.sType != vk_info.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO:
            raise AssertionError(f"unexpected InstanceInfo sType: {info.sType}")
        if not info.pApplicationInfo:
            raise AssertionError("InstanceInfo.pApplicationInfo is null")
        app = info.pApplicationInfo.contents
        if app.sType != vk_info.VK_STRUCTURE_TYPE_APPLICATION_INFO:
            raise AssertionError(f"unexpected AppInfo sType: {app.sType}")
        if app.apiVersion != vk_info.VK_API_VERSION_1_3:
            raise AssertionError(f"unexpected apiVersion: {app.apiVersion:#x}")
        if self.create_instance_result != 0:
            return self.create_instance_result
        ctypes.c_void_p.from_address(_ptr(p_inst)).value = self.INSTANCE_HANDLE
        return 0

    def _destroy_instance(self, inst, _p_alloc):
        self.destroyed_instances.append(_ptr(inst))

    def _enumerate_physical_devices(self, inst, p_count, p_devs):
        if _ptr(inst) != self.INSTANCE_HANDLE:
            raise AssertionError(f"unexpected instance handle: {_ptr(inst)!r}")
        self.enum_calls += 1
        if self.enumerate_hook is not None:
            return self.enumerate_hook(inst, p_count, p_devs)
        count_ptr = ctypes.c_uint32.from_address(_ptr(p_count))
        devs_addr = _ptr(p_devs)
        if devs_addr is None:
            count_ptr.value = len(self._device_handles)
            return 0
        cap = count_ptr.value
        written = min(cap, len(self._device_handles))
        if written:
            arr = (ctypes.c_void_p * written).from_address(devs_addr)
            for i in range(written):
                arr[i] = self._device_handles[i]
        count_ptr.value = written
        return vk_info.VK_INCOMPLETE if cap < len(self._device_handles) else 0

    def _enumerate_device_extension_properties(self, physical: ctypes.c_void_p, layer: None,
                                               p_count: object, p_entries: object) -> int:
        assert layer is None
        names = self._devices[_ptr(physical)].extension_names
        count = ctypes.c_uint32.from_address(_ptr(p_count))
        addr = _ptr(p_entries)
        if addr is None:
            count.value = len(names)
            return 0
        written = min(count.value, len(names))
        for i, name in enumerate(names[:written]):
            entry = addr + i * 260
            ctypes.memmove(entry, name + b"\0", len(name) + 1)
            ctypes.c_uint32.from_address(entry + 256).value = 1
        count.value = written
        return vk_info.VK_INCOMPLETE if written < len(names) else 0

    def _get_physical_device_properties2(self, phys_dev, p_props):
        handle = _ptr(phys_dev)
        self.props_calls.append(handle)
        dev = self._devices[handle]
        addr = _ptr(p_props)
        if ctypes.c_uint32.from_address(addr).value != vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2:
            raise AssertionError("missing VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 sType")

        base = addr + 16
        ctypes.c_uint32.from_address(base + vk_info.PROPS_API_VERSION).value = dev.api_version
        ctypes.c_uint32.from_address(base + vk_info.PROPS_DEVICE_TYPE).value = dev.device_type
        ctypes.c_uint32.from_address(base + vk_info.PROPS_VENDOR_ID).value = dev.vendor_id
        ctypes.c_uint32.from_address(base + vk_info.PROPS_DEVICE_ID).value = dev.device_id
        ctypes.c_uint32.from_address(base + vk_info.PROPS_MAX_PUSH_CONSTANTS_SIZE).value = dev.max_push_constants_size
        raw_name = dev.name.encode("utf-8") if isinstance(dev.name, str) else dev.name
        name_bytes = raw_name[:256].ljust(256, b"\x00")
        ctypes.memmove(base + vk_info.PROPS_DEVICE_NAME, name_bytes, 256)

        found_subgroup = False
        found_sg = False
        for stype, node_addr in self._iter_chain(addr):
            if stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES:
                found_subgroup = True
                subgroup = vk_info.SubgroupProperties.from_address(node_addr)
                subgroup.subgroupSize = dev.subgroup_size
                subgroup.supportedStages = dev.subgroup_supported_stages
            elif stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES:
                found_sg = True
                sg = vk_info.SubgroupSizeControl.from_address(node_addr)
                sg.minSubgroupSize = dev.min_subgroup_size
                sg.maxSubgroupSize = dev.max_subgroup_size
                sg.maxComputeWorkgroupSubgroups = dev.max_compute_workgroup_subgroups
                sg.requiredSubgroupSizeStages = dev.required_subgroup_size_stages
            elif stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES:
                ctypes.c_uint32.from_address(node_addr + 16).value = dev.driver_id
        if not found_subgroup:
            raise AssertionError("SubgroupProperties missing from vkGetPhysicalDeviceProperties2 pNext chain")
        if not found_sg:
            raise AssertionError("SubgroupSizeControl missing from vkGetPhysicalDeviceProperties2 pNext chain")

    def _get_physical_device_features2(self, phys_dev, p_features):
        handle = _ptr(phys_dev)
        self.features_calls.append(handle)
        dev = self._devices[handle]
        addr = _ptr(p_features)
        if ctypes.c_uint32.from_address(addr).value != vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
            raise AssertionError("missing VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 sType")

        seen_types = set()
        for stype, node_addr in self._iter_chain(addr):
            seen_types.add(stype)
            if stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F12_SEPARATE_DEPTH_STENCIL_LAYOUTS
                ).value = int(dev.separate_depth_stencil_layouts)
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F12_TIMELINE_SEMAPHORE
                ).value = int(dev.timeline_semaphore)
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F12_BUFFER_DEVICE_ADDRESS
                ).value = int(dev.buffer_device_address)
            elif stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F13_SUBGROUP_SIZE_CONTROL
                ).value = int(dev.subgroup_size_control)
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F13_SYNCHRONIZATION2
                ).value = int(dev.synchronization2)
                ctypes.c_uint32.from_address(
                    node_addr + vk_info.F13_DYNAMIC_RENDERING
                ).value = int(dev.dynamic_rendering)
            elif stype == 1000044003:  # VkPhysicalDeviceDynamicRenderingFeaturesKHR
                ctypes.c_uint32.from_address(node_addr + 16).value = int(dev.dynamic_rendering)
            elif stype == 1000314007:  # VkPhysicalDeviceSynchronization2FeaturesKHR
                ctypes.c_uint32.from_address(node_addr + 16).value = int(dev.synchronization2)
            elif stype == vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR:
                ctypes.c_uint32.from_address(node_addr + 16).value = int(dev.fragment_shader_barycentric)
        expected = {
            vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
        }
        if dev.api_version >= vk_info.VK_API_VERSION_1_3:
            expected.add(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
        else:
            expected.update((1000044003, 1000314007))
        if expected != seen_types:
            raise AssertionError(f"unexpected feature structs in pNext chain: {seen_types!r}")

    def _get_queue_family_properties(self, phys_dev, p_count, p_props):
        handle = _ptr(phys_dev)
        self.queue_calls.append(handle)
        dev = self._devices[handle]
        count_ptr = ctypes.c_uint32.from_address(_ptr(p_count))
        props_addr = _ptr(p_props)
        if props_addr is None:
            count_ptr.value = len(dev.queue_flags)
            return
        cap = count_ptr.value
        written = min(cap, len(dev.queue_flags))
        for i in range(written):
            entry = props_addr + i * vk_info.QUEUE_FAMILY_SIZE
            ctypes.c_uint32.from_address(entry).value = dev.queue_flags[i]
            ctypes.c_uint32.from_address(entry + 4).value = 1
        count_ptr.value = written


class StructHelpersTest(unittest.TestCase):
    def test_block_sets_stype_and_zero_fills_remainder(self):
        for size, stype in (
            (16, vk_info.VK_STRUCTURE_TYPE_APPLICATION_INFO),
            (vk_info.F13_SIZE, vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES),
            (16 + vk_info.PROPS_SIZE, vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2),
        ):
            with self.subTest(size=size, stype=stype):
                buf = vk_info.block(size, stype)
                self.assertEqual(len(buf), size)
                self.assertEqual(vk_info.u32(buf, 0), stype)
                self.assertEqual(bytes(buf)[4:], b"\x00" * (size - 4))

    def test_set_next_writes_target_address_at_offset_8(self):
        head = vk_info.block(vk_info.FEATURES2_SIZE, vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
        tail = vk_info.block(vk_info.F12_SIZE, vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
        ctypes.c_uint32.from_buffer(head, 16).value = 0xDEADBEEF

        vk_info.set_next(head, tail)

        self.assertEqual(vk_info.u32(head, 0), vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
        self.assertEqual(ctypes.c_void_p.from_buffer(head, 8).value, ctypes.addressof(tail))
        self.assertEqual(vk_info.u32(head, 16), 0xDEADBEEF)

        sg = vk_info.SubgroupSizeControl(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES)
        vk_info.set_next(tail, sg)
        self.assertEqual(ctypes.c_void_p.from_buffer(tail, 8).value, ctypes.addressof(sg))

    def test_u32_reads_unsigned_32_bit_values(self):
        buf = ctypes.create_string_buffer(24)
        for off, value in ((0, 0), (4, 1), (8, 0x12345678), (16, 0x80000000), (20, 0xFFFFFFFF)):
            ctypes.c_uint32.from_buffer(buf, off).value = value
            self.assertEqual(vk_info.u32(buf, off), value)




class ConstantsAndLayoutTest(unittest.TestCase):
    def test_vulkan_structure_types(self):
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_APPLICATION_INFO, 0)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 1)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, 1000059000)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, 1000059001)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, 1000094000)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, 51)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, 53)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, 1000225000)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, 1000196000)
        self.assertEqual(vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR, 1000203000)

    def test_shader_stage_bits(self):
        self.assertEqual(vk_info.VK_SHADER_STAGE_COMPUTE_BIT, 0x20)
        self.assertEqual(vk_info.VK_SHADER_STAGE_MESH_BIT_EXT, 0x80)
        self.assertEqual(vk_info.VK_SHADER_STAGE_COMPUTE_BIT & vk_info.VK_SHADER_STAGE_MESH_BIT_EXT, 0)

    def test_queue_and_device_flags(self):
        self.assertEqual(vk_info.VK_QUEUE_GRAPHICS_BIT, 1)
        self.assertEqual(vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 2)
        self.assertEqual(vk_info.VK_INCOMPLETE, 5)

    def test_driver_and_vendor_constants(self):
        self.assertEqual(vk_info.VK_DRIVER_ID_AMD_PROPRIETARY, 1)
        self.assertEqual(vk_info.VK_DRIVER_ID_MOLTENVK, 14)
        self.assertEqual(vk_info.AMD_VENDOR_ID, 0x1002)

    def test_api_version_1_3_encoding(self):
        expected = (1 << 22) | (3 << 12)
        self.assertEqual(vk_info.VK_API_VERSION_1_3, expected)
        self.assertGreater(vk_info.VK_API_VERSION_1_3, (1 << 22) | (2 << 12))

    def test_structure_sizes(self):
        self.assertEqual(vk_info.PROPS_SIZE, 824)
        self.assertEqual(vk_info.FEATURES2_SIZE, 240)
        self.assertEqual(vk_info.F12_SIZE, 208)
        self.assertEqual(vk_info.F13_SIZE, 80)
        self.assertEqual(vk_info.BARY_FEATURES_SIZE, 24)
        self.assertEqual(vk_info.DRIVER_PROPS_SIZE, 536)
        self.assertEqual(vk_info.QUEUE_FAMILY_SIZE, 24)

    def test_feature_and_property_offsets(self):
        self.assertEqual(vk_info.F12_SEPARATE_DEPTH_STENCIL_LAYOUTS, 16 + 4 * 35)
        self.assertEqual(vk_info.F12_TIMELINE_SEMAPHORE, 16 + 4 * 37)
        self.assertEqual(vk_info.F13_SUBGROUP_SIZE_CONTROL, 16 + 4 * 7)
        self.assertEqual(vk_info.F13_SYNCHRONIZATION2, 16 + 4 * 9)
        self.assertEqual(vk_info.F13_DYNAMIC_RENDERING, 16 + 4 * 12)
        self.assertEqual(vk_info.PROPS_API_VERSION, 0)
        self.assertEqual(vk_info.PROPS_VENDOR_ID, 8)
        self.assertEqual(vk_info.PROPS_DEVICE_ID, 12)
        self.assertEqual(vk_info.PROPS_DEVICE_TYPE, 16)
        self.assertEqual(vk_info.PROPS_DEVICE_NAME, 20)

    def test_ctypes_structure_layouts(self):
        sg = vk_info.SubgroupSizeControl(
            sType=vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
            pNext=None,
            minSubgroupSize=16,
            maxSubgroupSize=64,
            maxComputeWorkgroupSubgroups=128,
            requiredSubgroupSizeStages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        self.assertEqual(sg.sType, vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES)
        self.assertIsNone(sg.pNext)
        self.assertEqual(sg.minSubgroupSize, 16)
        self.assertEqual(sg.maxSubgroupSize, 64)
        self.assertEqual(sg.maxComputeWorkgroupSubgroups, 128)
        self.assertEqual(sg.requiredSubgroupSizeStages, vk_info.VK_SHADER_STAGE_COMPUTE_BIT)

        props = vk_info.SubgroupProperties(
            sType=vk_info.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
            pNext=None,
            subgroupSize=32,
            supportedStages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT,
            supportedOperations=0x1FF,
            quadOperationsInAllStages=1,
        )
        self.assertEqual(props.subgroupSize, 32)
        self.assertEqual(props.supportedStages, vk_info.VK_SHADER_STAGE_COMPUTE_BIT | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT)


class LoadVulkanTest(unittest.TestCase):
    def test_windows_loads_vulkan_1_dll(self):
        sentinel = object()
        with mock.patch.object(vk_info.os, "name", "nt"), \
                mock.patch.object(vk_info.ctypes, "CDLL", return_value=sentinel) as cdll:
            self.assertIs(vk_info.load_vulkan(), sentinel)
        cdll.assert_called_once_with("vulkan-1.dll")

    def test_windows_failure_lists_tried_library_name(self):
        with mock.patch.object(vk_info.os, "name", "nt"), \
                mock.patch.object(vk_info.ctypes, "CDLL", side_effect=OSError("not found")):
            with self.assertRaisesRegex(RuntimeError, r"no Vulkan loader found \(vulkan-1\.dll\)"):
                vk_info.load_vulkan()

    def test_macos_loads_moltenvk(self):
        with mock.patch.object(vk_info.os, "name", "posix"), mock.patch.object(vk_info.sys, "platform", "darwin"), \
                mock.patch.dict(vk_info.os.environ, {"LEONIDA_VULKAN_LOADER": ""}):
            with mock.patch.object(vk_info.ctypes, "CDLL") as cdll:
                self.assertIs(vk_info.load_vulkan(), cdll.return_value)
                cdll.assert_called_once_with("libMoltenVK.dylib")
            sentinel = object()
            with mock.patch.object(vk_info.ctypes, "CDLL", side_effect=[OSError("missing"), sentinel]) as cdll:
                self.assertIs(vk_info.load_vulkan(), sentinel)
                self.assertEqual(cdll.call_args_list, [mock.call("libMoltenVK.dylib"),
                                                       mock.call("/opt/homebrew/lib/libMoltenVK.dylib")])

    def test_macos_uses_only_the_explicit_loader(self):
        path = "/custom/libMoltenVK.dylib"
        with mock.patch.object(vk_info.os, "name", "posix"), mock.patch.object(vk_info.sys, "platform", "darwin"), \
                mock.patch.dict(vk_info.os.environ, {"LEONIDA_VULKAN_LOADER": path}):
            with mock.patch.object(vk_info.ctypes, "CDLL") as cdll:
                self.assertIs(vk_info.load_vulkan(), cdll.return_value)
                cdll.assert_called_once_with(path)
            with mock.patch.object(vk_info.ctypes, "CDLL", side_effect=OSError("missing")) as cdll:
                with self.assertRaisesRegex(RuntimeError, r"no Vulkan loader found \(/custom/libMoltenVK\.dylib\)"):
                    vk_info.load_vulkan()
                cdll.assert_called_once_with(path)

    def test_posix_prefers_so_and_falls_back_to_dylib(self):
        sentinel = object()
        with mock.patch.object(vk_info.os, "name", "posix"), mock.patch.object(vk_info.sys, "platform", "linux"), \
                mock.patch.object(vk_info.ctypes, "CDLL", return_value=sentinel) as cdll:
            self.assertIs(vk_info.load_vulkan(), sentinel)
        cdll.assert_called_once_with("libvulkan.so.1")

        def fake_cdll(name):
            if name == "libvulkan.so.1":
                raise OSError("missing linux loader")
            return sentinel

        with mock.patch.object(vk_info.os, "name", "posix"), mock.patch.object(vk_info.sys, "platform", "linux"), \
                mock.patch.object(vk_info.ctypes, "CDLL", side_effect=fake_cdll) as cdll:
            self.assertIs(vk_info.load_vulkan(), sentinel)
        self.assertEqual([c.args[0] for c in cdll.call_args_list], ["libvulkan.so.1", "libvulkan.1.dylib"])

    def test_posix_failure_lists_all_tried_names(self):
        with mock.patch.object(vk_info.os, "name", "posix"), mock.patch.object(vk_info.sys, "platform", "linux"), \
                mock.patch.object(vk_info.ctypes, "CDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(
                RuntimeError, r"no Vulkan loader found \(libvulkan\.so\.1, libvulkan\.1\.dylib\)"
            ):
                vk_info.load_vulkan()


class PhysicalDevicesTest(unittest.TestCase):
    def _make_enum_vk(self, steps):
        """Build a minimal vk object whose vkEnumeratePhysicalDevices replays (status, handles) steps."""
        calls = []
        it = iter(steps)

        def enumerate_devices(inst, p_count, p_devs):
            calls.append((_ptr(inst), _ptr(p_devs)))
            status, handles = next(it)
            count_ptr = ctypes.c_uint32.from_address(_ptr(p_count))
            devs_addr = _ptr(p_devs)
            if devs_addr is not None and handles:
                arr = (ctypes.c_void_p * len(handles)).from_address(devs_addr)
                for i, h in enumerate(handles):
                    arr[i] = h
            count_ptr.value = len(handles) if isinstance(handles, (list, tuple)) else handles
            return status

        vk = mock.Mock()
        vk.vkEnumeratePhysicalDevices.side_effect = enumerate_devices
        return vk, calls

    def test_empty_count_returns_empty_list_in_single_call(self):
        vk, calls = self._make_enum_vk([(0, 0)])
        devs = vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
        self.assertEqual(devs, [])
        self.assertEqual(calls, [(FakeVulkan.INSTANCE_HANDLE, None)])

    def test_returns_handles_and_respects_shrinking_second_count(self):
        vk, calls = self._make_enum_vk([(0, 3), (0, [0xA1, 0xB2])])
        devs = vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
        self.assertEqual(devs, [0xA1, 0xB2])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0][1])
        self.assertIsNotNone(calls[1][1])

    def test_count_failure_raises_runtime_error(self):
        vk, calls = self._make_enum_vk([(-3, 0)])
        with self.assertRaisesRegex(RuntimeError, r"vkEnumeratePhysicalDevices count failed \(-3\)"):
            vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
        self.assertEqual(len(calls), 1)

    def test_vk_incomplete_triggers_recount_and_succeeds(self):
        for retries, steps, expected in (
            (1, [(0, 1), (vk_info.VK_INCOMPLETE, [0x11]), (0, 2), (0, [0x11, 0x22])], [0x11, 0x22]),
            (2, [
                (0, 1), (vk_info.VK_INCOMPLETE, [0x11]),
                (0, 2), (vk_info.VK_INCOMPLETE, [0x11, 0x22]),
                (0, 3), (0, [0x11, 0x22, 0x33]),
            ], [0x11, 0x22, 0x33]),
        ):
            with self.subTest(retries=retries):
                vk, calls = self._make_enum_vk(steps)
                devs = vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
                self.assertEqual(devs, expected)
                self.assertEqual(len(calls), len(steps))

    def test_repeated_vk_incomplete_three_times_raises(self):
        steps = [
            (0, 1), (vk_info.VK_INCOMPLETE, [0x11]),
            (0, 2), (vk_info.VK_INCOMPLETE, [0x11, 0x22]),
            (0, 3), (vk_info.VK_INCOMPLETE, [0x11, 0x22, 0x33]),
        ]
        vk, calls = self._make_enum_vk(steps)
        with self.assertRaisesRegex(RuntimeError, "repeatedly returned VK_INCOMPLETE"):
            vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
        self.assertEqual(len(calls), 6)

    def test_non_incomplete_second_call_error_raises_immediately(self):
        vk, calls = self._make_enum_vk([(0, 2), (-1, [])])
        with self.assertRaisesRegex(RuntimeError, r"vkEnumeratePhysicalDevices failed \(-1\)"):
            vk_info.physical_devices(vk, ctypes.c_void_p(FakeVulkan.INSTANCE_HANDLE))
        self.assertEqual(len(calls), 2)


class QueryTest(unittest.TestCase):
    def setUp(self):
        # A LEONIDA_VK_DEVICE in the caller's environment would narrow the fake devices too.
        env = mock.patch.dict(vk_info.os.environ)
        env.start()
        self.addCleanup(env.stop)
        vk_info.os.environ.pop("LEONIDA_VK_DEVICE", None)

    def _run_query(self, *devices):
        vk = FakeVulkan(devices=devices)
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            return vk_info.query(), vk

    def test_32bit_python_guard_raises_before_loading_vulkan(self):
        for maxsize in (2**31 - 1, 2**32):
            with self.subTest(maxsize=maxsize), \
                    mock.patch.object(vk_info.sys, "maxsize", maxsize), \
                    mock.patch.object(vk_info, "load_vulkan") as load:
                with self.assertRaisesRegex(RuntimeError, "64-bit"):
                    vk_info.query()
                load.assert_not_called()

    def test_vk_create_instance_failure_raises_without_destroying(self):
        vk = FakeVulkan(create_instance_result=-9)
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            with self.assertRaisesRegex(RuntimeError, r"vkCreateInstance failed \(-9\)"):
                vk_info.query()
        self.assertEqual(vk.destroyed_instances, [])

    def test_no_physical_devices_destroys_instance(self):
        vk = FakeVulkan(devices=())
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            with self.assertRaisesRegex(RuntimeError, "no Vulkan device"):
                vk_info.query()
        self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_enumeration_failure_destroys_instance(self):
        vk = FakeVulkan(enumerate_hook=lambda _inst, _count, _devs: -3)
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            with self.assertRaisesRegex(RuntimeError, r"vkEnumeratePhysicalDevices count failed \(-3\)"):
                vk_info.query()
        self.assertEqual(vk.props_calls, [])
        self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_no_suitable_device_destroys_instance(self):
        vk = FakeVulkan(devices=(FakeDevice(api_version=(1 << 22) | (2 << 12)),))
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            with self.assertRaisesRegex(RuntimeError, "no Vulkan 1.3 GPU"):
                vk_info.query()
        self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_device_filtering_rejects_missing_requirements(self):
        rejected_cases = [
            ("vulkan_1_2", FakeDevice(name="Old", api_version=(1 << 22) | (2 << 12))),
            ("below_1_3_by_one", FakeDevice(name="Almost13", api_version=vk_info.VK_API_VERSION_1_3 - 1)),
            ("no_dynamic_rendering", FakeDevice(name="NoDyn", dynamic_rendering=False)),
            ("no_synchronization2", FakeDevice(name="NoSync2", synchronization2=False)),
            ("no_buffer_device_address", FakeDevice(name="NoBDA", buffer_device_address=False)),
            ("no_timeline_semaphore", FakeDevice(name="NoTimeline", timeline_semaphore=False)),
            ("no_separate_ds", FakeDevice(name="NoSepDS", separate_depth_stencil_layouts=False)),
            ("zero_queue_families", FakeDevice(name="NoQueues", queue_flags=())),
            ("compute_transfer_only", FakeDevice(name="NoGraphics", queue_flags=(0x02, 0x04))),
        ]
        fallback = FakeDevice(name="Fallback OK")

        for label, bad_dev in rejected_cases:
            with self.subTest(case=label):
                vk_only_bad = FakeVulkan(devices=(bad_dev,))
                with mock.patch.object(vk_info, "load_vulkan", return_value=vk_only_bad):
                    with self.assertRaisesRegex(RuntimeError, "no Vulkan 1.3 GPU"):
                        vk_info.query()
                self.assertEqual(vk_only_bad.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])
                if bad_dev.api_version < vk_info.VK_API_VERSION_1_3:
                    self.assertEqual(vk_only_bad.features_calls, [])
                    self.assertEqual(vk_only_bad.queue_calls, [])

                res, vk_with_fallback = self._run_query(bad_dev, fallback)
                self.assertEqual(res["device"], "Fallback OK")
                self.assertEqual(vk_with_fallback.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_vulkan_1_2_khr_features_match_the_runtime_device_choice(self) -> None:
        extensions = (b"VK_KHR_dynamic_rendering", b"VK_KHR_synchronization2")
        fallback = FakeDevice(name="Vulkan 1.3 integrated GPU", device_type=1)
        for names, dynamic, sync in (
            (extensions, True, True),
            (extensions[:1], True, True),
            (extensions[1:], True, True),
            (extensions, False, True),
            (extensions, True, False),
        ):
            with self.subTest(names=names, dynamic=dynamic, sync=sync):
                candidate = FakeDevice(name="Vulkan 1.2 discrete GPU", api_version=(1 << 22) | (2 << 12),
                                       extension_names=names, dynamic_rendering=dynamic, synchronization2=sync)
                result, _ = self._run_query(fallback, candidate)
                suitable = names == extensions and dynamic and sync
                self.assertEqual(result["device"], candidate.name if suitable else fallback.name)
                if suitable:
                    self.assertFalse(result["subgroup_size_control"])
                    self.assertFalse(result["native64"])

    def test_accepts_graphics_bit_on_non_zero_queue_family(self):
        dev = FakeDevice(
            name="MultiQueue GPU",
            queue_flags=(0x02, 0x04, 0x02 | vk_info.VK_QUEUE_GRAPHICS_BIT),
        )
        res, _vk = self._run_query(dev)
        self.assertEqual(res["device"], "MultiQueue GPU")
        self.assertTrue(res["native64"])

    def test_device_selection_rule(self):
        igpu1 = FakeDevice(name="iGPU-1", device_type=1)
        igpu2 = FakeDevice(name="iGPU-2", device_type=1)
        cpu = FakeDevice(name="CPU", device_type=4)
        dgpu1 = FakeDevice(name="dGPU-1", device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        dgpu2 = FakeDevice(name="dGPU-2", device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        bad_dgpu = FakeDevice(
            name="Bad-dGPU",
            device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            dynamic_rendering=False,
        )
        dozen = FakeDevice(name="Microsoft Direct3D12 (dGPU-1)",
                           device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                           vendor_id=vk_info.MICROSOFT_VENDOR_ID)

        cases = [
            ("discrete_beats_earlier_integrated", (igpu1, dgpu1), "dGPU-1"),
            ("discrete_beats_later_integrated", (dgpu1, igpu1), "dGPU-1"),
            ("first_discrete_wins_a_tie", (igpu1, dgpu1, dgpu2, igpu2), "dGPU-1"),
            ("integrated_beats_an_earlier_cpu", (cpu, igpu1), "iGPU-1"),
            ("native_beats_a_later_dozen", (dgpu1, dozen), "dGPU-1"),
            ("native_beats_an_earlier_dozen", (dozen, dgpu1), "dGPU-1"),
            ("native_integrated_beats_dozen", (dozen, igpu1), "iGPU-1"),
            ("dozen_when_it_is_all_there_is", (dozen, bad_dgpu), "Microsoft Direct3D12 (dGPU-1)"),
            ("unsuitable_later_discrete_does_not_override", (dgpu1, bad_dgpu), "dGPU-1"),
            ("first_non_discrete_when_no_discrete", (igpu1, cpu, igpu2), "iGPU-1"),
            ("first_suitable_non_discrete_after_rejected", (bad_dgpu, igpu1, igpu2), "iGPU-1"),
        ]
        for label, devs, expected_name in cases:
            with self.subTest(case=label):
                res, vk = self._run_query(*devs)
                self.assertEqual(res["device"], expected_name)
                self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_leonida_vk_device_filters_case_insensitively_and_fails_on_no_match(self):
        igpu = FakeDevice(
            name="AMD Radeon(TM) 780M Graphics",
            device_type=1,
            subgroup_size=64,
            min_subgroup_size=32,
            max_subgroup_size=64,
        )
        dgpu = FakeDevice(
            name="NVIDIA GeForce RTX 4070",
            device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            subgroup_size=32,
            min_subgroup_size=32,
            max_subgroup_size=32,
        )
        other_dgpu = FakeDevice(
            name="AMD Radeon RX 7900 XTX",
            device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            subgroup_size=64,
            min_subgroup_size=32,
            max_subgroup_size=64,
        )
        old_match = FakeDevice(
            name="Legacy Radeon 1.2",
            api_version=(1 << 22) | (2 << 12),
        )

        # Unset or empty selects the last discrete GPU as usual.
        for empty in ("",):
            with mock.patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": empty}):
                res, _ = self._run_query(igpu, dgpu)
                self.assertEqual(res["device"], "NVIDIA GeForce RTX 4070")
                self.assertFalse(res["native64"])

        # Case-insensitive substring can select the integrated GPU over a later discrete GPU.
        for filt in ("780m", "RADEON(tm) 780M", "amd radeon"):
            with self.subTest(filter=filt), mock.patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": filt}):
                res, vk = self._run_query(igpu, dgpu)
                self.assertEqual(res["device"], "AMD Radeon(TM) 780M Graphics")
                self.assertTrue(res["native64"])
                # Non-matching discrete GPU was skipped before feature/queue queries.
                self.assertEqual(vk.features_calls, [vk._device_handles[0]])
                self.assertEqual(vk.queue_calls, [vk._device_handles[0], vk._device_handles[0]])

        # Among multiple matching devices, the runtime's discrete/last-discrete rule still applies.
        with mock.patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": "radeon"}):
            res, _ = self._run_query(igpu, dgpu, other_dgpu)
            self.assertEqual(res["device"], "AMD Radeon RX 7900 XTX")
        with mock.patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": "geforce"}):
            res, _ = self._run_query(igpu, dgpu, other_dgpu)
            self.assertEqual(res["device"], "NVIDIA GeForce RTX 4070")

        # A filter that matches no device (or only an unsuitable device) raises RuntimeError and destroys the instance.
        for filt, devs in (("nomatch", (igpu, dgpu)), ("legacy", (old_match, dgpu))):
            with self.subTest(no_match=filt), mock.patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": filt}):
                vk = FakeVulkan(devices=devs)
                with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
                    with self.assertRaisesRegex(RuntimeError, rf"LEONIDA_VK_DEVICE={filt}"):
                        vk_info.query()
                self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])

    def test_stage_bitmask_with_additional_shader_stages(self):
        extra_stages = (
            vk_info.VK_SHADER_STAGE_COMPUTE_BIT
            | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT
            | 0x01
            | 0x10
        )
        dev = FakeDevice(
            required_subgroup_size_stages=extra_stages,
            subgroup_supported_stages=extra_stages,
        )
        res, _vk = self._run_query(dev)
        self.assertTrue(res["compute_in_required_stages"])
        self.assertTrue(res["mesh_in_required_stages"])
        self.assertTrue(res["native64"])

    def test_native64_conditions_and_boundaries(self):
        both = vk_info.VK_SHADER_STAGE_COMPUTE_BIT | vk_info.VK_SHADER_STAGE_MESH_BIT_EXT
        cases = [
            ("typical_32_64_both_stages", True, 32, 64, both, True, True, True, 32),
            ("exact_64_64_boundary", True, 64, 64, both, True, True, True, 64),
            ("range_16_128_includes_64", True, 16, 128, both, True, True, True, 32),
            ("extra_stage_bits_allowed", True, 32, 64, both | 0x10 | 0x40, True, True, True, 32),
            ("missing_subgroup_size_control", False, 32, 64, both, True, True, False, 32),
            ("max_subgroup_63_too_small", True, 32, 63, both, True, True, False, 32),
            ("max_subgroup_32_too_small", True, 32, 32, both, True, True, False, 32),
            ("min_subgroup_65_too_large", True, 65, 128, both, True, True, False, 128),
            ("missing_compute_stage_bit", True, 32, 64, vk_info.VK_SHADER_STAGE_MESH_BIT_EXT, False, True, False, 32),
            ("missing_mesh_stage_bit", True, 32, 64, vk_info.VK_SHADER_STAGE_COMPUTE_BIT, True, False, False, 32),
            ("no_required_stage_bits", True, 32, 64, 0, False, False, False, 32),
        ]
        for label, sgc, lo, hi, stages, want_comp, want_mesh, want_native64, subgroup_size in cases:
            with self.subTest(case=label):
                dev = FakeDevice(
                    name=f"GPU-{label}",
                    subgroup_size_control=sgc,
                    subgroup_size=subgroup_size,
                    min_subgroup_size=lo,
                    max_subgroup_size=hi,
                    required_subgroup_size_stages=stages,
                )
                res, _vk = self._run_query(dev)
                self.assertEqual(res, {
                    "device": f"GPU-{label}",
                    "subgroup_size": subgroup_size,
                    "min_subgroup_size": lo,
                    "max_subgroup_size": hi,
                    "subgroup_size_control": sgc,
                    "compute_in_required_stages": want_comp,
                    "mesh_in_required_stages": want_mesh,
                    "compute_in_supported_stages": True,
                    "mesh_in_supported_stages": True,
                    "native64_compute": want_native64 or label == "missing_mesh_stage_bit",
                    "native64": want_native64,
                    "per_vertex_inputs": True,
                    "barycentric_fallback": False,
                    "max_push_constants_size": 256,
                    "vendor_id": 0,
                    "device_id": 0,
                    "driver_id": 1,
                    "rdna1_amd_windows": False,
                    "rdna2_amd_windows": False,
                })

    def test_max_push_constants_size(self):
        """limits.maxPushConstantsSize is read from VkPhysicalDeviceProperties (offset 328 on 64-bit)."""
        for limit in (128, 256, 4096):
            with self.subTest(limit=limit):
                res, _vk = self._run_query(FakeDevice(max_push_constants_size=limit))
                self.assertEqual(res["max_push_constants_size"], limit)

    def test_per_vertex_inputs(self):
        """PerVertexKHR needs fragmentShaderBarycentric, and MoltenVK can't compile it to MSL either way."""
        for label, bary, driver, want in (("amd", True, 1, True), ("no_barycentrics", False, 1, False),
                                          ("moltenvk", True, vk_info.VK_DRIVER_ID_MOLTENVK, False)):
            with self.subTest(case=label):
                res, _vk = self._run_query(FakeDevice(fragment_shader_barycentric=bary, driver_id=driver))
                self.assertIs(res["per_vertex_inputs"], want)

    def test_barycentric_fallback(self):
        for label, driver, want in (("anv", vk_info.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, True),
                                    ("radv", 3, False), ("intel_windows", 5, False)):
            with self.subTest(case=label):
                res, _vk = self._run_query(FakeDevice(driver_id=driver))
                self.assertIs(res["barycentric_fallback"], want)
        self.assertEqual(vk_info.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, 6)

    def test_barycentric_fallback_rejects_an_intel_windows_fake(self):
        """Intel Windows uses its Windows driver ID, not ANV; the fake must carry Intel's vendor ID."""
        res, _vk = self._run_query(FakeDevice(name="Intel Arc", vendor_id=0x8086, device_id=0xE20B,
                                               driver_id=5))
        self.assertFalse(res["barycentric_fallback"])
        self.assertEqual(vk_info.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, 6)

    def test_barycentric_fallback_gfx90c_amd_windows(self):
        """#5933: gfx90c iGPUs (amdgpu's CHIP_RENOIR IDs) under AMD's proprietary driver (driver ID 1) take the
        fallback; RADV on the same chip, other GCN 5 parts, RDNA iGPUs and other vendors do not. Literal IDs, so a
        changed constant or table fails here."""
        cases = (("vega8_barcelo", 0x1002, 0x15E7, 1, True), ("renoir", 0x1002, 0x1636, 1, True),
                 ("cezanne", 0x1002, 0x1638, 1, True), ("lucienne", 0x1002, 0x164C, 1, True),
                 ("vega8_radv", 0x1002, 0x15E7, 3, False), ("vega8_amdvlk_open", 0x1002, 0x15E7, 2, False),
                 ("raven_gfx902", 0x1002, 0x15DD, 1, False), ("picasso_gfx902", 0x1002, 0x15D8, 1, False),
                 ("vega10_card", 0x1002, 0x687F, 1, False), ("vega20_card", 0x1002, 0x66AF, 1, False),
                 ("raphael_rdna2_igpu", 0x1002, 0x164E, 1, False), ("rdna3_780m", 0x1002, 0x15BF, 1, False),
                 ("other_vendor_same_id", 0x10DE, 0x15E7, 1, False),
                 ("intel_windows", 0x8086, 0x15E7, 5, False), ("anv", 0x8086, 0xE20B, 6, True))
        for label, vendor, device, driver, want in cases:
            with self.subTest(case=label):
                self.assertIs(vk_info.barycentric_fallback(vendor, device, driver), want)
                self.assertIs(vk_info.gfx90c_amd_windows(vendor, device, driver), want and driver == 1)
                res, _vk = self._run_query(FakeDevice(vendor_id=vendor, device_id=device, driver_id=driver))
                self.assertEqual((res["vendor_id"], res["device_id"], res["driver_id"]), (vendor, device, driver))
                self.assertIs(res["barycentric_fallback"], want)

    def test_barycentric_fallback_answer_is_for_the_runtime_gpu(self):
        """A Vega 8 iGPU next to an NVIDIA card: the runtime takes the discrete card, so no fallback; alone, it does."""
        vega8 = FakeDevice(name="AMD Radeon(TM) Graphics", device_type=1, vendor_id=0x1002, device_id=0x15E7)
        nvidia = FakeDevice(name="NVIDIA GeForce", vendor_id=0x10DE, device_id=0x2882, driver_id=4,
                            min_subgroup_size=32, max_subgroup_size=32)
        for devs in ((vega8, nvidia), (nvidia, vega8)):
            with self.subTest(order=[d.name for d in devs]):
                res, _vk = self._run_query(*devs)
                self.assertEqual(res["device"], "NVIDIA GeForce")
                self.assertIs(res["barycentric_fallback"], False)
        res, _vk = self._run_query(vega8)
        self.assertIs(res["barycentric_fallback"], True)

    def test_stage_caps_preserve_graphics_stage_bits(self):
        supported = 0x01 | 0x08 | 0x10 | 0x20
        required = 0x10 | 0x20
        device = FakeDevice(subgroup_supported_stages=supported, required_subgroup_size_stages=required,
                            max_compute_workgroup_subgroups=17)
        vk = FakeVulkan(devices=(device,))
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk):
            result = vk_info.query(stage_caps=True)
        self.assertEqual(result["supported_subgroup_stages"], supported)
        self.assertEqual(result["required_subgroup_size_stages"], required)
        self.assertEqual(result["max_compute_workgroup_subgroups"], 17)

    def test_rdna2_amd_windows(self):
        """#857: RDNA2 by PCI device ID, and only under AMD's proprietary driver (RADV takes the same modules)."""
        amd, radv = vk_info.VK_DRIVER_ID_AMD_PROPRIETARY, 3
        cases = (("rx6700", 0x1002, 0x73DF, amd, True), ("rx6800", 0x1002, 0x73BF, amd, True),
                 ("navi24", 0x1002, 0x743F, amd, True), ("granite_ridge_igpu", 0x1002, 0x13C0, amd, True),
                 ("raphael_igpu", 0x1002, 0x164E, amd, True), ("rx6700_radv", 0x1002, 0x73DF, radv, False),
                 ("rdna3_7900xtx", 0x1002, 0x744C, amd, False), ("rdna4_9070xt", 0x1002, 0x7550, amd, False),
                 ("rdna3_780m", 0x1002, 0x15BF, amd, False), ("nvidia_same_id", 0x10DE, 0x73DF, 4, False))
        for label, vendor, device, driver, want in cases:
            with self.subTest(case=label):
                self.assertIs(vk_info.rdna2_amd_windows(vendor, device, driver), want)
                res, _vk = self._run_query(FakeDevice(vendor_id=vendor, device_id=device, driver_id=driver))
                self.assertEqual((res["vendor_id"], res["device_id"], res["driver_id"]), (vendor, device, driver))
                self.assertIs(res["rdna2_amd_windows"], want)

    def test_rdna1_amd_windows(self):
        amd = vk_info.VK_DRIVER_ID_AMD_PROPRIETARY
        cases = ((0x1002, 0x731F, amd, True), (0x1002, 0x7340, amd, True),
                 (0x1002, 0x7360, amd, True), (0x1002, 0x731F, 3, False),
                 (0x1002, 0x73DF, amd, False), (0x10DE, 0x731F, amd, False))
        for vendor, device, driver, expected in cases:
            with self.subTest(vendor=vendor, device=device, driver=driver):
                res, _vk = self._run_query(FakeDevice(vendor_id=vendor, device_id=device, driver_id=driver))
                self.assertIs(res["rdna1_amd_windows"], expected)

    def test_rdna2_answer_is_for_the_runtime_gpu(self):
        """An Intel iGPU next to an RX 6700: the runtime takes the discrete card, and so does the answer."""
        intel = FakeDevice(name="Intel iGPU", device_type=1, vendor_id=0x8086, device_id=0xA780, driver_id=5,
                           min_subgroup_size=8, max_subgroup_size=32)
        rx = FakeDevice(name="AMD Radeon RX 6700", vendor_id=0x1002, device_id=0x73DF)
        for devs in ((intel, rx), (rx, intel)):
            with self.subTest(order=[d.name for d in devs]):
                res, _vk = self._run_query(*devs)
                self.assertEqual(res["device"], "AMD Radeon RX 6700")
                self.assertTrue(res["rdna2_amd_windows"])
                self.assertTrue(res["native64"])

    def test_native64_uses_default_size_without_required_mesh_stage(self):
        dev = FakeDevice(
            subgroup_size=64,
            required_subgroup_size_stages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        res, _vk = self._run_query(dev)
        self.assertEqual(res["subgroup_size"], 64)
        self.assertTrue(res["compute_in_supported_stages"])
        self.assertTrue(res["mesh_in_supported_stages"])
        self.assertTrue(res["compute_in_required_stages"])
        self.assertFalse(res["mesh_in_required_stages"])
        self.assertTrue(res["native64"])

    def test_native64_default_size_requires_both_supported_stages(self):
        dev = FakeDevice(
            subgroup_size=64,
            subgroup_supported_stages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
            required_subgroup_size_stages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        res, _vk = self._run_query(dev)
        self.assertFalse(res["mesh_in_supported_stages"])
        self.assertTrue(res["native64_compute"])
        self.assertFalse(res["native64"])

    def test_native64_required_size_requires_both_supported_stages(self):
        dev = FakeDevice(
            subgroup_supported_stages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        res, _vk = self._run_query(dev)
        self.assertTrue(res["mesh_in_required_stages"])
        self.assertFalse(res["mesh_in_supported_stages"])
        self.assertTrue(res["native64_compute"])
        self.assertFalse(res["native64"])

    def test_native64_compute_requires_compute_subgroup_support(self):
        dev = FakeDevice(
            subgroup_size=64,
            subgroup_supported_stages=vk_info.VK_SHADER_STAGE_MESH_BIT_EXT,
            required_subgroup_size_stages=vk_info.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        res, _vk = self._run_query(dev)
        self.assertFalse(res["native64_compute"])

    def test_device_name_stops_at_nul_and_replaces_invalid_utf8(self):
        res_nul, _ = self._run_query(FakeDevice(name=b"RDNA2\x00garbage_after_nul"))
        self.assertEqual(res_nul["device"], "RDNA2")

        res_bad_utf8, _ = self._run_query(FakeDevice(name=b"GPU-\xff-Test\x00"))
        self.assertEqual(res_bad_utf8["device"], "GPU-\ufffd-Test")

    def test_query_integrates_with_load_vulkan_cdll(self):
        vk = FakeVulkan(devices=(FakeDevice(name="Via CDLL"),))
        with mock.patch.object(vk_info.os, "name", "nt"), \
                mock.patch.object(vk_info.ctypes, "CDLL", return_value=vk) as cdll:
            res = vk_info.query()
        cdll.assert_called_once_with("vulkan-1.dll")
        self.assertEqual(res["device"], "Via CDLL")
        self.assertTrue(res["native64"])
        self.assertEqual(vk.destroyed_instances, [FakeVulkan.INSTANCE_HANDLE])


class MainCliTest(unittest.TestCase):
    def setUp(self):
        # A LEONIDA_VK_DEVICE in the caller's environment would narrow the fake devices too.
        env = mock.patch.dict(vk_info.os.environ)
        env.start()
        self.addCleanup(env.stop)
        vk_info.os.environ.pop("LEONIDA_VK_DEVICE", None)

    def test_main_exits_0_and_prints_single_line_json_on_stdout(self):
        vk = FakeVulkan(devices=(FakeDevice(name="CLI GPU"),))
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = vk_info.main()
        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["device"], "CLI GPU")
        self.assertTrue(payload["native64"])

    def test_main_exits_1_and_prints_json_error_on_stdout_not_stderr(self):
        vk = FakeVulkan(devices=())
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(vk_info, "load_vulkan", return_value=vk), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = vk_info.main()
        self.assertEqual(code, 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {"error": "RuntimeError: no Vulkan device"})

        out2, err2 = io.StringIO(), io.StringIO()
        with mock.patch.object(vk_info, "query", side_effect=OSError("loader broken")), \
                contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
            code2 = vk_info.main()
        self.assertEqual(code2, 1)
        self.assertEqual(err2.getvalue(), "")
        self.assertEqual(json.loads(out2.getvalue().strip()), {"error": "OSError: loader broken"})


if __name__ == "__main__":
    unittest.main()
