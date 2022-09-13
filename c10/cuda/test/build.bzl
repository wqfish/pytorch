def define_targets(rules):
    rules.cc_test(
        name = "test",
        srcs = [
            "impl/CUDATest.cpp",
            "impl/CUDAAssertionsTest.cu",
        ],
        local_defines = [
            "TORCH_USE_CUDA_DSA",
        ],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/cuda",
        ],
        target_compatible_with = rules.requires_cuda_enabled(),
    )
