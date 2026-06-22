#!/usr/bin/env python
# SConstruct  —  M-ABD GDExtension 빌드 스크립트
#
# 사용법:
#   scons target=template_debug    (debug 빌드)
#   scons target=template_release  (release 빌드)
#   scons target=template_debug platform=windows
#   scons target=template_debug platform=linux
#
# godot-cpp 경로는 GODOT_CPP_PATH 환경변수 또는 아래 기본값 사용.
import os, sys

env = SConscript("godot-cpp/SConstruct")

env.Append(CPPPATH=["src/"])

sources = Glob("src/*.cpp")

# ── 최적화 플래그 ─────────────────────────────────────────────
if env["target"] == "template_release":
    env.Append(CCFLAGS=["/O3", "/DNDEBUG", "/FS"])
else:
    env.Append(CCFLAGS=["/O2", "/Zi", "/FS"])
    
# C++17 (DynMat 등 structured bindings 사용)
env.Append(CXXFLAGS=["/std:c++17"])

# ── 라이브러리 출력 ───────────────────────────────────────────
library = env.SharedLibrary(
    "demo/bin/libmabdphysics{}{}".format(
        env["suffix"], env["SHLIBSUFFIX"]
    ),
    source=sources,
)

Default(library)