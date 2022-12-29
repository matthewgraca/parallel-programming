cc_library(
    name = "amdahls_law",
    srcs = ["amdahls_law.cc"],
    hdrs = ["amdahls_law.h"],
)

cc_test(
    name = "amdahls_law_unittest",
    size = "small",
    srcs = ["amdahls_law_unittest.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        ":amdahls_law"
        ],
)
