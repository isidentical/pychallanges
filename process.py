from __future__ import annotations

import ast
import dataclasses
import json
import shutil
import sys
from argparse import ArgumentParser
from contextlib import redirect_stdout
from dataclasses import dataclass
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Iterator, List, Type

EXAMPLE_RUNS = 3
METADATA_TREE = "__metadata"


class ASTState(Enum):
    EXTRACT = auto()


def _evaluate_class(node: ast.ClassDef) -> Type:
    wrapped_module = ast.Module(body=[node], type_ignores=[])
    evaluation_bytecode = compile(wrapped_module, __file__, "exec")

    tmp_namespace = {"ast": ast}
    exec(evaluation_bytecode, tmp_namespace)
    return tmp_namespace.get(node.name)


class ChallangeSolver(ast.NodeTransformer):
    def visit_Constant(self, node):
        if self.is_answer_target(node):
            # we have to unwrap the current node
            # from the ast.Expr
            wrapped_node = next(self.answers)
            return wrapped_node.value
        else:
            return self.generic_visit(node)

    def visit_Expr(self, node):
        if self.is_answer_target(node.value):
            return next(self.answers)
        else:
            return self.generic_visit(node)

    def solve(self, tree: ast.AST, answers: List[str]) -> ast.AST:
        self.answers = self._prepare_answer_nodes(answers)
        return self.visit(tree)

    def _prepare_answer_nodes(self, answers: List[str]) -> Iterator[ast.AST]:
        for answer in answers:
            tree = ast.parse(answer)
            if len(tree.body) > 1:
                raise ValueError(
                    "Every answer should correspond to a single AST unit"
                )
            yield tree.body[0]

    @staticmethod
    def is_answer_target(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is Ellipsis


def run_challange(tree: ast.AST, runs: int) -> List[str]:
    outputs = []
    bytecode = compile(tree, __file__, "exec")
    for run in range(runs):
        namespace = {}
        with redirect_stdout(StringIO()) as buffer:
            exec(bytecode, namespace)
        outputs.append(buffer.getvalue())
    return outputs


@dataclass
class Category:
    name: str
    items: List[str]


@dataclass
class Challange:
    name: str
    source: str

    rules: List[str]
    answers: List[str]
    outputs: List[str]

    @classmethod
    def from_raw_source(cls, name: str, source: str) -> Challange:
        tree = ast.parse(source)
        if (
            isinstance(tree.body[0], ast.ClassDef)
            and tree.body[0].name == METADATA_TREE
        ):
            raw_def = tree.body.pop(0)
            metadata = _evaluate_class(raw_def)
        else:
            raise ValueError(
                f"Missing {METADATA_TREE} class at the top of the file!"
            )

        solved_tree = ChallangeSolver().solve(tree, metadata.answers)
        example_outputs = run_challange(solved_tree, EXAMPLE_RUNS)
        challange_source_lines = ast._splitlines_no_ff(source)
        challange_source = "".join(
            challange_source_lines[raw_def.end_lineno :]
        )
        return cls(
            name=name,
            source=challange_source,
            rules=metadata.rules,
            answers=metadata.answers,
            outputs=example_outputs,
        )


def generate_api(input_dir: Path, output_dir: Path) -> None:
    shutil.rmtree(output_dir)
    output_dir.mkdir()

    collections = {
        category.stem: category.glob("**/*.py")
        for category in input_dir.iterdir()
    }

    for category, challanges in collections.items():
        category_parent = output_dir / category
        category_parent.mkdir()
        challange_names = []
        for challange in challanges:
            challange_name = challange.stem
            with open(
                (category_parent / challange_name).with_suffix(".json"), "w"
            ) as file:
                data = Challange.from_raw_source(
                    name=challange_name, source=challange.read_text()
                )
                json.dump(dataclasses.asdict(data), file, indent=4)
            challange_names.append(challange_name)

        with open(category_parent / "all.json", "w") as file:
            data = Category(category, challange_names)
            json.dump(dataclasses.asdict(data), file, indent=4)

    with open(output_dir / "all.json", "w") as file:
        data = Category("main", list(collections.keys()))
        json.dump(dataclasses.asdict(data), file, indent=4)


def main(argv: List[str]) -> int:
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("source/"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/"))

    options = parser.parse_args(argv)
    generate_api(**vars(options))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
