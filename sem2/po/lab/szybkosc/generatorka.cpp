#include<bits/stdc++.h>
using namespace std;

int rand(int a, int b){
  return a + rand() % (b - a + 1);
}

int main(int argc, char *argv[]){
  srand(atoi(argv[1]));

  int n = rand(3, 100000);
  cout << n << '\n';
  for (int i = 0; i < n; i++)
    cout << rand(0, 1000000) << ' ' << rand(1, 1000000) << '\n';
}
