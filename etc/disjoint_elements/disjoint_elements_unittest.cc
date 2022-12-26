#include "disjoint_elements.h"
#include "gtest/gtest.h"

TEST(DisjointElementsTest, NoDisjoint){
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(count(a, b), 0);
}
