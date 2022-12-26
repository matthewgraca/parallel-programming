#include "disjoint_elements.h"
#include <unordered_map>

// counts the number of disjoint elements between two int arrays, same size
int sequential_count(int a[], int b[], int n){
  int count = 0;
  // store b into a hashmap
  std::unordered_map<int, int> umap;
  for (int i = 0; i < n; i++){
    umap.insert({b[i], i});
  }
  // count the number of elements in a that are contained in b
  for (int i = 0; i < n; i++){
    if (!umap.contains(a[i]))
      count += 2; // (1)
  }
  return count;
}
/* (1) we count twice since if a doesn't appear in b, 
 b also doesn't appear in a if both sets are the same size 
 (otherwise, we'd store both a and b in separate hashmaps 
 and count elements of a not in b, then elements in b not in a)
*/
