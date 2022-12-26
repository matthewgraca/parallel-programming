#include "disjoint_elements.h"
#include "gtest/gtest.h"

TEST(DisjointElementsTest, EmptyIsNoDisjoint){
  int n = 0;
  int a[] = {};
  int b[] = {};
  EXPECT_EQ(sequential_count(a, b, n), 0);
}

TEST(DisjointElementsTest, NoDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(sequential_count(a, b, n), 0);
}

TEST(DisjointElementsTest, OneDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 9};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(sequential_count(a, b, n), 2);
}

TEST(DisjointElementsTest, AllDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {6, 7, 8, 9, 10, 11};
  EXPECT_EQ(sequential_count(a, b, n), 12);
}
