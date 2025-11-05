# deepcoder_data.py  (fixed)
from typing import Dict, Any, List
import os, glob, json, subprocess, tempfile, textwrap, sys

from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets

class DeepCoderAdapter:
    """
    Adapter for PrimeIntellect/deepcoder-gold-standard-solutions.

    Expected columns per dataset_info.json:
      - 'prompt' (str)
      - 'gold_standard_solution' (str)
      - 'verification_info' (str, often a JSON string with tests or judge info)
    """
    def __init__(self, path_or_repo: str):
        self.split = self._load_any(path_or_repo)

        # Map correct keys for this dataset (see dataset_info.json)
        self.prompt_key = "prompt"
        self.solution_key = "gold_standard_solution"
        self.verif_key   = "verification_info"

    # ---- loaders ----

    def _is_saved_dataset_dir(self, p: str) -> bool:
        return os.path.exists(os.path.join(p, "dataset_dict.json")) or \
               os.path.exists(os.path.join(p, "state.json"))

    def _try_load_arrow_shards(self, p: str):
        """
        Load a Dataset by concatenating .arrow shards (works on HF cache dirs like yours).
        """
        shard_paths = sorted(glob.glob(os.path.join(p, "*train-*.arrow")))
        if not shard_paths:
            return None
        parts = [Dataset.from_file(sp) for sp in shard_paths]
        return concatenate_datasets(parts)

    def _load_any(self, p: str):
        # Case 1: a directory produced by save_to_disk()
        if os.path.isdir(p) and self._is_saved_dataset_dir(p):
            return load_from_disk(p)

        # Case 2: HF cache folder with .arrow shards (your current case)
        if os.path.isdir(p):
            ds = self._try_load_arrow_shards(p)
            if ds is not None:
                return ds

        # Case 3: repo id or fallback to local cache of the repo id
        # First try local-only to avoid re-downloading; if that fails, allow remote.
        try:
            return load_dataset("PrimeIntellect/deepcoder-gold-standard-solutions",
                                split="train", local_files_only=True)
        except Exception:
            return load_dataset("PrimeIntellect/deepcoder-gold-standard-solutions",
                                split="train")

    # ---- API ----

    def __len__(self):
        return len(self.split)

    def get_prompt(self, idx: int) -> str:
        return str(self.split[idx][self.prompt_key])

    def get_reference(self, idx: int) -> Dict[str, Any]:
        rec = self.split[idx]
        ref = {"solution": rec.get(self.solution_key)}
        vi  = rec.get(self.verif_key)
        # verification_info is a string; try to parse JSON if present
        if isinstance(vi, str) and vi.strip():
            try:
                ref["verification_info"] = json.loads(vi)
            except Exception:
                ref["verification_info"] = vi
        return ref

# -------------------------
# Verifier (unchanged exec sandbox, now understands verification_info)
# -------------------------

def run_python_code_with_tests(code: str, tests: List[Dict[str, Any]], timeout_sec: int = 2) -> bool:
    code = textwrap.dedent(code)
    with tempfile.TemporaryDirectory() as tmp:
        main_py = os.path.join(tmp, "main.py")
        with open(main_py, "w", encoding="utf-8") as f:
            f.write(code)
        driver = os.path.join(tmp, "driver.py")
        driver_code = """
import importlib, json, sys
m = importlib.import_module("main")
ok = True
tests = json.loads(sys.stdin.read())
for t in tests:
    func = getattr(m, t.get("entry", "solve"), None)
    if func is None:
        ok = False; break
    out = func(*t.get("input", []))
    if out != t.get("output"):
        ok = False; break
print("OK" if ok else "FAIL")
"""
        with open(driver, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(driver_code))
        p = subprocess.Popen([sys.executable, driver],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=tmp)
        out, err = p.communicate(input=json.dumps(tests).encode("utf-8"), timeout=timeout_sec)
        return out.decode("utf-8").strip() == "OK"

def deepcoder_reward(generated: str, reference: Dict[str, Any]) -> float:
    """
    Graded reward in [0,1]:
      - +0.10 if file parses and defines the expected entry function (default 'solve').
      - +0.20 if at least one test executes without crashing.
      - +0.70 * (passed / total tests) for full pass fraction.
    Falls back to exact string match when no verification info exists.
    """
    vi = reference.get("verification_info")
    try:
        if isinstance(vi, str):
            vi = json.loads(vi)
    except Exception:
        vi = None

    entry = "solve"
    tests = None
    if isinstance(vi, dict):
        entry = vi.get("entry", entry)
        tests = vi.get("tests") if isinstance(vi.get("tests"), list) else None

    bonus = 0.0

    try:
        import ast
        tree = ast.parse(generated)
        has_entry = any(isinstance(node, ast.FunctionDef) and node.name == entry for node in tree.body)
        if has_entry:
            bonus += 0.10
        else:
            # No entry function defined; give partial credit only if exact match fallback passes.
            raise ValueError("missing entry function")
    except Exception:
        sol = reference.get("solution")
        return 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0

    if tests:
        try:
            # Probe run using the first test to award execution bonus.
            initial_ok = run_python_code_with_tests(generated, tests[:1])
            if initial_ok:
                bonus += 0.20
            passed = 0
            for test_case in tests:
                try:
                    if run_python_code_with_tests(generated, [test_case]):
                        passed += 1
                except Exception:
                    pass
            frac = passed / max(1, len(tests))
            return min(1.0, bonus + 0.70 * frac)
        except Exception:
            return bonus

    sol = reference.get("solution")
    match = 1.0 if isinstance(sol, str) and generated.strip() == sol.strip() else 0.0
    return max(bonus, match)
