#include "disjoint_elements.h"
#include "gtest/gtest.h"

/******************************************************************************
 * test sequential count
 *****************************************************************************/

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

TEST(DisjointElementsSequentialTest, AllDisjointLargeDataset){
  int n = 10000;
  int a[n];
  int b[n];
  for (int i = 0; i < n; i++){
    a[i] = i*2;
    b[i] = i*2+1;
  }
  EXPECT_EQ(sequential_count(a, b, n), 20000);
}

TEST(DisjointElementsSequentialTest, SomeDisjointLargeDataset){
  int n = 10000;
  int a[n], b[n];
  std::srand(0);
  for (int i = 0; i < n; i++){
    a[i] = rand() % 10000;
    b[i] = rand() % 10000;
  }
  EXPECT_EQ(sequential_count(a, b, n), 7396);
}

TEST(DisjointElementsSequentialTest, SomeDisjointiGigaDataset){
  int n = 1000000;
  int a[n], b[n];
  std::srand(0);
  for (int i = 0; i < n; i++){
    a[i] = rand() % 1000000;
    b[i] = rand() % 1000000;
  }
  EXPECT_EQ(sequential_count(a, b, n), 736036);
}

/******************************************************************************
 * test parallel count
 *****************************************************************************/

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

TEST(DisjointElementsParallelTest, AllDisjointLargeDataset){
  int n = 10000;
  int a[n];
  int b[n];
  for (int i = 0; i < n; i++){
    a[i] = i*2;
    b[i] = i*2+1;
  }
  EXPECT_EQ(parallel_count(a, b, n), 20000);
}

TEST(DisjointElementsParallelTest, SomeDisjointLargeDataset){
  int n = 10000;
  int a[n], b[n];
  std::srand(0);
  for (int i = 0; i < n; i++){
    a[i] = rand() % 10000;
    b[i] = rand() % 10000;
  }
  EXPECT_EQ(parallel_count(a, b, n), 7396);
}

TEST(DisjointElementsParallelTest, SomeDisjointiGigaDataset){
  int n = 1000000;
  int a[n], b[n];
  std::srand(0);
  for (int i = 0; i < n; i++){
    a[i] = rand() % 1000000;
    b[i] = rand() % 1000000;
  }
  EXPECT_EQ(parallel_count(a, b, n), 736036);
}

/******************************************************************************
 * test naive count
 *****************************************************************************/

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

TEST(DisjointElementsNaiveTest, AllDisjointLargeDataset){
  int n = 10000;
  int a[n];
  int b[n];
  for (int i = 0; i < n; i++){
    a[i] = i*2;
    b[i] = i*2+1;
  }
  EXPECT_EQ(naive_parallel_count(a, b, n), 20000);
}

TEST(DisjointElementsNaiveTest, SomeDisjointLargeDataset){
  int n = 10000;
  int a[n], b[n];
  std::srand(0);
  for (int i = 0; i < n; i++){
    a[i] = rand() % 10000;
    b[i] = rand() % 10000;
  }
  EXPECT_EQ(naive_parallel_count(a, b, n), 7396);
}
