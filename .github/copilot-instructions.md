# CUDA-Practice repository instructions

This repository is a modular CUDA practice project built with CMake.

Project characteristics:
- Root build system uses CMake.
- Each top-level topic directory is mostly independent.
- Prioritize correctness over aggressive optimization.
- Preserve current project layout and executable structure.

Build:
1. mkdir build && cd build
2. cmake ..
3. cmake --build . --parallel 8

Validation:
- Keep changes minimal and local.
- Do not refactor unrelated modules.
- Do not rename directories or reorganize the repository.
- When possible, verify GPU results against CPU logic or expected output.
- Prefer fixing correctness, safety, and build issues first.

Review priorities:
1. wrong results
2. out-of-bounds accesses
3. synchronization bugs
4. invalid CUDA API usage
5. missing error checks
6. build failures
7. only then style or maintainability

Scope rules:
- Do not modify many top-level directories in one task.
- One PR should focus on one module or one bug category.
- Avoid changing CMake unless necessary for build correctness.

Output expectations:
- explain changed files
- explain the root cause
- explain how the fix was validated
- mention remaining risks
