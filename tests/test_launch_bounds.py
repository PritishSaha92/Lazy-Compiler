"""Launch bounds describe the prescribed launch, including auxiliary threads."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT)]
from test_builtin_aux import cases as builtin_cases
from ptx_backend import Compiler, CompileError
from tools.check_compiler import assemble


class LaunchBoundsTests(unittest.TestCase):
    def test_header_only_and_prescribed_dimensions(self):
        # Both 3-D launch and an auxiliary-role CTA are covered by this fixture
        # collection. Removing the directive must recover exactly the baseline.
        for graph, meta, _ in builtin_cases():
            launch = meta['kernels'][0]
            with self.subTest(case=meta['id']):
                before = Compiler(graph, launch, target='sm_90', launch_bounds=False).compile()
                after = Compiler(graph, launch, target='sm_90').compile()
                expected = '.maxntid ' + ', '.join(map(str, launch['block'])) + '\n'
                self.assertEqual(after.count(expected), 1)
                self.assertEqual(after.replace(expected, ''), before)
                if shutil.which('ptxas'):
                    assemble(after, 'sm_90')

    def test_invalid_block_contract_is_diagnosed(self):
        graph, problem, _ = next(iter(builtin_cases()))
        for block in ([], [32], [32, 1, 0], [32, True, 1], [32, 1.0, 1], [1024, 2, 1]):
            meta = deepcopy(problem['kernels'][0])
            meta['block'] = block
            with self.subTest(block=block), self.assertRaisesRegex(CompileError, 'block must contain'):
                Compiler(graph, meta, target='sm_90').compile()


if __name__ == '__main__':
    unittest.main()
