#include<bits/stdc++.h>
using namespace std;

template <typename T> T getczary(){//magia!
  long long ujemna = false, znak = getchar_unlocked();
  T wynik = (T)0;
  while(!isdigit(znak)){
    if(znak == '-')
      ujemna = true;
    znak = getchar_unlocked();
  }
  while(isdigit(znak)){
    wynik *= 10;
    wynik += znak - '0';
    znak = getchar_unlocked();
  }
  if(ujemna)
    wynik *= -1;
  return wynik;
}
 
struct Point{
  long long x, y;
  Point(){}
  Point(const long long &_x, const long long &_y): x(_x), y(_y){}
  Point operator -(const Point &a) const{ return {x - a.x, y - a.y}; }
  Point operator +(const Point &a) const{ return {x + a.x, y + a.y}; }
  Point operator *(const long long &a) const{ return {x * a, y * a}; }
  
  long long cross_product(Point a) const{ return x * a.y - y * a.x; }
  long long dot_product(Point a) const{ return x * a.x + y * a.y; }
  
  int side(const Point &a, const Point &b) const{
    long long orientation = (b - a).cross_product(*this - a);
    return orientation == 0 ? 0 : (orientation > 0 ? 1 : 2);
  }
  
  bool operator<(const Point &a) const{ return x < a.x || (x == a.x && y < a.y); }
};
Point read_point(){
  long long x = getczary<long long>(), y = getczary<long long>();
  return Point(x, y);
}
long double shoelace(vector<Point> &p){ // Polygon area
  long long res = 0;
  for(int n = p.size(), i = 0, j = n - 1;i < n;j = i++)
    res += p[i].cross_product(p[j]);
  return (long double)abs(res) / 2;
}
 
void convex_hull(vector<Point> p, vector<Point> &ow){
  sort(p.begin(), p.end());
  for(auto pt : p){
    while(ow.size() >= 2 && pt.side(ow[ow.size() - 2], ow.back()) == 1) ow.pop_back();
    ow.emplace_back(pt);
  }
  ow.pop_back(); int k = ow.size();
  reverse(p.begin(), p.end());
  for(auto pt : p){
    while(ow.size() >= k + 2 && pt.side(ow[ow.size() - 2], ow.back()) == 1) ow.pop_back();
    ow.emplace_back(pt);
  }
  ow.pop_back();
}
int main(){
  vector<Point> p, ow;
  for(int n = getczary<int>();n;n--)
    p.emplace_back(read_point());
  convex_hull(p, ow);
  printf("%lu\n", ow.size());
}
