#include "amdahls_law.h"
#include "gtest/gtest.h"

TEST(NoParallel, Test){
  EXPECT_EQ(no_parallel(), 1);
}

TEST(QuarterParallel, Test){
  EXPECT_EQ(quarter_parallel(), 1);
}

TEST(HalfParallel, Test){
  EXPECT_EQ(half_parallel(), 1);
}

TEST(ThreeQuartersParallel, Test){
  EXPECT_EQ(three_quarters_parallel(), 1);
}

TEST(FullParallel, Test){
  EXPECT_EQ(full_parallel(), 1);
}
