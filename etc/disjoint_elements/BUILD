cc_library(
    name = "disjoint_elements",
    srcs = ["disjoint_elements.cc"],
    hdrs = ["disjoint_elements.h"],
)

cc_test(
    name = "disjoint_elements_unittest",
    size = "small",
    srcs = ["disjoint_elements_unittest.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        ":disjoint_elements"
        ],
)
