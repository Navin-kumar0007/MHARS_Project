import ast
from pathlib import Path


ROOT = Path("/Users/navin/MHARS_Project")
SOURCE = ROOT / "tools" / "fix_mhars_paper_v2.py"


def main():
    tree = ast.parse(SOURCE.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_figures")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "FIG_DIR": ROOT / "generated_v2_figures",
        "RESULTS_DIR": ROOT / "results",
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    namespace["generate_figures"]()


if __name__ == "__main__":
    main()
