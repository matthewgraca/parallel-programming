#include "amdahls_law.h"
#include "gtest/gtest.h"

TEST(Test, Test){
  EXPECT_EQ(no_parallel(), 0);
}
