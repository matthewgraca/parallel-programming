#include "disjoint_elements.h"
#include "gtest/gtest.h"

// test sequential count
TEST(DisjointElementsSequentialTest, EmptyIsNoDisjoint){
  int n = 0;
  int a[] = {};
  int b[] = {};
  EXPECT_EQ(sequential_count(a, b, n), 0);
}

TEST(DisjointElementsSequentialTest, NoDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(sequential_count(a, b, n), 0);
}

TEST(DisjointElementsSequentialTest, OneDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 9};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(sequential_count(a, b, n), 2);
}

TEST(DisjointElementsSequentialTest, AllDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {6, 7, 8, 9, 10, 11};
  EXPECT_EQ(sequential_count(a, b, n), 12);
}

// test parallel count
TEST(DisjointElementsParallelTest, EmptyIsNoDisjoint){
  int n = 0;
  int a[] = {};
  int b[] = {};
  EXPECT_EQ(parallel_count(a, b, n), 0);
}

TEST(DisjointElementsParallelTest, NoDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(parallel_count(a, b, n), 0);
}

TEST(DisjointElementsParallelTest, OneDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 9};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(parallel_count(a, b, n), 2);
}

TEST(DisjointElementsParallelTest, AllDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {6, 7, 8, 9, 10, 11};
  EXPECT_EQ(parallel_count(a, b, n), 12);
}

// test naive count
TEST(DisjointElementsNaiveTest, EmptyIsNoDisjoint){
  int n = 0;
  int a[] = {};
  int b[] = {};
  EXPECT_EQ(naive_parallel_count(a, b, n), 0);
}

TEST(DisjointElementsNaiveTest, NoDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(naive_parallel_count(a, b, n), 0);
}

TEST(DisjointElementsNaiveTest, OneDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 9};
  int b[] = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(naive_parallel_count(a, b, n), 2);
}

TEST(DisjointElementsNaiveTest, AllDisjoint){
  int n = 6;
  int a[] = {0, 1, 2, 3, 4, 5};
  int b[] = {6, 7, 8, 9, 10, 11};
  EXPECT_EQ(naive_parallel_count(a, b, n), 12);
}


