#!/usr/bin/env bash
# Cross-compile Fast DDS for Android (static libs), matching ROS 2 Humble's versions.
#
#     ANDROID_HOME=~/Android/Sdk ./build_fastdds.sh [abi ...]   (default: arm64-v8a x86_64)
#
# Sources go to native/src, installs to native/fastdds/<abi>; both are gitignored.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/native/src"
OUT="$HERE/native/fastdds"
ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
NDK="${ANDROID_NDK:-$(ls -d "$ANDROID_HOME"/ndk/* | sort -V | tail -1)}"
CMAKE="$(ls -d "$ANDROID_HOME"/cmake/*/bin/cmake 2>/dev/null | sort -V | tail -1 || true)"
CMAKE="${CMAKE:-cmake}"
API=26
ABIS=("${@:-arm64-v8a x86_64}")
ABIS=(${ABIS[@]})

mkdir -p "$SRC"
clone() { [ -d "$SRC/$2" ] || git clone -q --depth 1 -b "$3" "$1" "$SRC/$2"; }
clone https://github.com/eProsima/Fast-CDR.git fastcdr v1.0.29
clone https://github.com/foonathan/memory.git foonathan_memory v0.7-3
clone https://github.com/eProsima/Fast-DDS.git fastdds v2.6.11
git -C "$SRC/fastdds" submodule update --init --depth 1 thirdparty/asio thirdparty/tinyxml2
# libc++'s system_clock counts microseconds, so adding raw nanoseconds does not compile.
sed -i 's/+= std::chrono::nanoseconds(\(.*\));/+= std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(\1));/' \
    "$SRC/fastdds/thirdparty/filewatch/FileWatch.hpp"

for ABI in "${ABIS[@]}"; do
    PREFIX="$OUT/$ABI"
    COMMON=(
        -G Ninja
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
        -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_shared
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"
        -DCMAKE_FIND_ROOT_PATH="$PREFIX"
    )
    build() {
        local name=$1; shift
        "$CMAKE" -S "$SRC/$name" -B "$HERE/native/build/$ABI/$name" "${COMMON[@]}" "$@"
        "$CMAKE" --build "$HERE/native/build/$ABI/$name" --target install -j"$(nproc)"
    }
    build fastcdr
    build foonathan_memory -DFOONATHAN_MEMORY_BUILD_EXAMPLES=OFF \
        -DFOONATHAN_MEMORY_BUILD_TESTS=OFF -DFOONATHAN_MEMORY_BUILD_TOOLS=OFF
    build fastdds -DTHIRDPARTY=FORCE -DTHIRDPARTY_fastcdr=OFF -DTHIRDPARTY_Asio=FORCE \
        -DTHIRDPARTY_TinyXML2=FORCE -DCOMPILE_TOOLS=OFF -DCOMPILE_EXAMPLES=OFF \
        -DSM_RUN_RESULT=1 -DSM_RUN_RESULT__TRYRUN_OUTPUT= \
        -DBUILD_DOCUMENTATION=OFF -DSECURITY=OFF -DSQLITE3_SUPPORT=OFF -DSHM_TRANSPORT_DEFAULT=OFF \
        -Dfastcdr_DIR="$PREFIX/lib/cmake/fastcdr" \
        -Dfoonathan_memory_DIR="$PREFIX/lib/foonathan_memory/cmake"
done
