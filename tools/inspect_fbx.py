"""Осмотр FBX-модели грузовика: меши, габариты, единицы, ориентация."""

import sys
from pathlib import Path

import numpy as np
import assimp_py

sys.stdout.reconfigure(encoding="utf-8")

PATH = Path("models/CargoTruck.fbx")
flags = assimp_py.Process_Triangulate | assimp_py.Process_PreTransformVertices
scene = assimp_py.import_file(str(PATH), flags)

print(f"файл: {PATH} ({PATH.stat().st_size} байт)")
print(f"мешей: {len(scene.meshes)}, материалов: {len(scene.materials)}")

parts = []
for i, mesh in enumerate(scene.meshes):
    v = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    if v.size == 0:
        continue
    parts.append(v)
    print(f"  mesh[{i}] '{mesh.name}': вершин {len(v)}, "
          f"треугольников {mesh.num_faces}, "
          f"габарит {np.round(v.max(0) - v.min(0), 3)}")

P = np.vstack(parts)
lo, hi = P.min(axis=0), P.max(axis=0)
size = hi - lo
print(f"\nобщий bbox (единицы файла): min={np.round(lo, 3)}, max={np.round(hi, 3)}")
print(f"размеры X,Y,Z = {np.round(size, 3)}, центр={np.round((lo + hi) / 2, 3)}")
print(f"самая длинная ось: {'XYZ'[int(np.argmax(size))]} "
      f"({size.max():.3f}), самая короткая: {'XYZ'[int(np.argmin(size))]} "
      f"({size.min():.3f})")


def walk(node, depth=0):
    m = np.asarray(node.transformation, dtype=float).reshape(4, 4)
    t = m[:3, 3]
    print("  " * depth + f"- {node.name}: t={np.round(t, 3)}")
    for child in node.children:
        walk(child, depth + 1)


print("\nиерархия узлов:")
walk(scene.root_node)
