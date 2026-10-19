"""Content identity for parsing algorithms, separate from lossless encoding.

Only the two reviewed portrait-region codecs are normalized. Everything else
remains in the AST, including detector, model, dtype and support calculations.
This helper neither imports a model runtime nor reads model weights.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path


SCHEMA = "portrait_regions_algorithm.v1"
_LEVEL_NAME = "NPZ_COMPRESSION_LEVEL"
_WRITER_NAME = "save_npz"
_WRITER_SOURCE = '''
def save_npz(path: Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=NPZ_COMPRESSION_LEVEL) as archive:
        for name, values in arrays.items():
            if not isinstance(name, str) or "/" in name or "\\\\" in name:
                raise ValueError("NPZ member names must be simple strings")
            with archive.open(name + ".npy", mode="w") as member:
                np.lib.format.write_array(member, np.asanyarray(values), allow_pickle=False)
'''


class _WithoutDocstrings(ast.NodeTransformer):
    def _body(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    visit_Module = _body
    visit_FunctionDef = _body
    visit_AsyncFunctionDef = _body
    visit_ClassDef = _body


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _compression_level(node):
    return (isinstance(node, ast.Constant) and type(node.value) is int
            and 0 <= node.value <= 9)


def _review_png_call(node):
    """The untouched review ROI is the sole approved PNG normalization."""
    return (isinstance(node.func, ast.Attribute) and node.func.attr == "save"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "crop"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Div)
            and isinstance(node.args[0].left, ast.Name) and node.args[0].left.id == "directory"
            and isinstance(node.args[0].right, ast.Constant)
            and node.args[0].right.value == "source-roi.png")


def _known_npz_writer(tree):
    """Accept the exact reviewed streaming writer, never an arbitrary helper.

    Removing a writer definition is safe only while its implementation and
    every use of its codec globals match the known lossless form. Unknown
    expressions or a change inside write_array stay in the algorithm identity.
    """
    writers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == _WRITER_NAME]
    levels = [node for node in tree.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == _LEVEL_NAME]
    if len(writers) != 1 or len(levels) != 1 or not _compression_level(levels[0].value):
        return None
    writer, level = writers[0], levels[0]
    expected = _WithoutDocstrings().visit(ast.parse(_WRITER_SOURCE)).body[0]
    if _dump(writer) != _dump(expected):
        return None
    inside_writer = {id(node) for node in ast.walk(writer)}
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    imported_zipfile = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"globals", "locals", "eval", "exec", "__import__"}):
            # Indirect access can turn an encoding constant into an algorithm
            # input. An unreviewed dynamic module keeps its full AST instead.
            return None
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == _WRITER_NAME and node is not writer):
            return None
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zipfile":
                    if alias.asname is not None or node not in tree.body:
                        return None
                    imported_zipfile = True
        if not isinstance(node, ast.Name):
            continue
        if node.id == "zipfile" and id(node) not in inside_writer:
            return None
        if node.id == _LEVEL_NAME and id(node) not in inside_writer and node is not level.targets[0]:
            return None
        if node.id == _WRITER_NAME:
            parent = parents.get(id(node))
            if not (isinstance(parent, ast.Call) and parent.func is node):
                return None
    return (writer, level) if imported_zipfile else None


class _LosslessCodecs(ast.NodeTransformer):
    def __init__(self, known_npz):
        self.known_npz = known_npz

    def visit_Call(self, node):
        self.generic_visit(node)
        if _review_png_call(node):
            # Omission selects Pillow's lossless default. Dynamic expressions
            # and any unreviewed option retain their exact AST.
            node.keywords = [keyword for keyword in node.keywords
                             if not (keyword.arg == "compress_level"
                                     and _compression_level(keyword.value))]
        if self.known_npz and isinstance(node.func, ast.Name) and node.func.id == _WRITER_NAME:
            node.func = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                      attr="savez_compressed", ctx=ast.Load())
        return node


def normalized_regions_ast(source: str | bytes) -> str:
    """Return the full parsing AST after the strict, reviewed normalizations."""
    if isinstance(source, bytes):
        source = source.decode("utf-8-sig")
    tree = _WithoutDocstrings().visit(ast.parse(source))
    known_npz = _known_npz_writer(tree)
    if known_npz:
        writer, level = known_npz
        tree.body = [node for node in tree.body if node is not writer and node is not level]
        for node in list(tree.body):
            if isinstance(node, ast.Import):
                node.names = [alias for alias in node.names if alias.name != "zipfile"]
                if not node.names:
                    tree.body.remove(node)
    tree = _LosslessCodecs(bool(known_npz)).visit(tree)
    return _dump(tree)


@lru_cache(maxsize=1)
def _normalizer_sha256():
    # A change to these normalization rules gets a fresh cache identity too.
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def regions_algorithm_sha256(source: str | bytes) -> str:
    payload = "\n".join((SCHEMA, _normalizer_sha256(), normalized_regions_ast(source)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _file_identity(path, size, modified_ns, created_ns):
    source = Path(path).read_bytes()
    return {"schema": SCHEMA, "algorithm_sha256": regions_algorithm_sha256(source),
            "producer_sha256": hashlib.sha256(source).hexdigest()}


def regions_code_identity(path: str | Path) -> dict:
    """Read a small parser source once per file version; no asset/model scan."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(_file_identity(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
