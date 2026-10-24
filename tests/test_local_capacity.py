"""CPU-only tests: the CUDA library is replaced before driver construction."""

import ctypes as ct
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_compiler as checker


class FakeCuda:
    def __init__(self, default, optin, status=0, static=0):
        self.default, self.optin, self.status, self.static = (
            default,
            optin,
            status,
            static,
        )
        self.optins = []
        self.launches = 0
        self.module_loads = 0

    def cuDeviceGetAttribute(self, value, attribute, device):
        if attribute.value == 8:
            value._obj.value = self.default
            return 0
        if attribute.value == 97:
            value._obj.value = self.optin
            return self.status
        raise AssertionError(attribute.value)

    def cuModuleLoadData(self, module, source):
        self.module_loads += 1
        module._obj.value = 1
        return 0

    def cuModuleGetFunction(self, function, module, name):
        function._obj.value = 2
        return 0

    def cuFuncGetAttribute(self, value, attribute, function):
        assert attribute.value == 1
        value._obj.value = self.static
        return 0

    def cuFuncSetAttribute(self, function, attribute, value):
        self.optins.append((attribute.value, value.value))
        return 0

    def cuLaunchKernel(self, *args):
        self.launches += 1
        return 0

    def cuCtxSynchronize(self):
        return 0


def driver(fake):
    # __init__ loads nvcuda.dll and creates a context. It is never called.
    result = checker.CudaDriver.__new__(checker.CudaDriver)
    result.lib = fake
    result.device = ct.c_int(0)
    result.module = ct.c_void_p()
    result.arch = "sm_75"
    result.name = "CPU mock, not a real GPU"
    result.allocations = []
    return result


def request(bytes_):
    return {
        "ptx": ".version 8.0\n.target sm_89\n.address_size 64\n",
        "adapt_target": True,
        "entry": "mock",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "smem_bytes": bytes_,
        "args": [],
    }


class CapacityQueryTests(unittest.TestCase):
    def test_pre_ampere_optin_is_decided_by_device_limit(self):
        for dynamic in (49284, 65536):
            with self.subTest(dynamic=dynamic):
                fake = FakeCuda(49152, 65536)
                result = driver(fake).run(request(dynamic))
                self.assertTrue(result["ok"])
                self.assertEqual(fake.launches, 1)
                self.assertEqual(fake.optins, [(8, dynamic)])

    def test_default_memory_needs_no_optin(self):
        for dynamic in (0, 49152):
            with self.subTest(dynamic=dynamic):
                fake = FakeCuda(49152, 65536)
                self.assertTrue(driver(fake).run(request(dynamic))["ok"])
                self.assertEqual(fake.optins, [])

    def test_actual_limit_and_missing_optin_remain_unsupported(self):
        for optin, status, dynamic in (
            (49152, 0, 49284),
            (65536, 0, 65540),
            (0, 1, 49284),
        ):
            with self.subTest(optin=optin, status=status, dynamic=dynamic):
                fake = FakeCuda(49152, optin, status=status)
                with self.assertRaises(checker.UnsupportedLocal):
                    driver(fake).run(request(dynamic))
                self.assertEqual(fake.launches, 0)
                self.assertEqual(fake.module_loads, 0)
        fake = FakeCuda(49152, 0, status=1)
        self.assertTrue(driver(fake).run(request(49152))["ok"])
        self.assertEqual(fake.optins, [])

    def test_static_plus_dynamic_respects_total_limit(self):
        fake = FakeCuda(49152, 65536, static=256)
        self.assertTrue(driver(fake).run(request(65024))["ok"])
        self.assertEqual(fake.optins, [(8, 65024)])
        fake = FakeCuda(49152, 65536, static=256)
        with self.assertRaises(checker.UnsupportedLocal):
            driver(fake).run(request(65408))
        self.assertEqual(fake.launches, 0)

    def test_unexpected_attribute_error_is_not_hidden(self):
        fake = FakeCuda(49152, 0, status=3)
        with self.assertRaisesRegex(RuntimeError, "CUDA 3"):
            driver(fake).run(request(0))
        self.assertEqual(fake.launches, 0)
        self.assertEqual(fake.module_loads, 0)


if __name__ == "__main__":
    unittest.main()
