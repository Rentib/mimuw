#include <iostream>
#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>

using umap_t = std::unordered_map<int, int>;
using uset_t = std::unordered_set<int>;
using top7_t =
  std::set<std::pair<int, int>, decltype([](const std::pair<int, int> &a,
                                            const std::pair<int, int> &b) {
    return a.first < b.first || (a.first == b.first && a.second > b.second);
  })>;

enum QueryType { kVote, kNew, kTop, kIgnore, kError };

namespace {

const std::regex e_num("[^\\s]+", std::regex_constants::optimize);

int MAX;
uset_t dropped;      // <song>
umap_t votes_count;  // [song] = votes
umap_t points_count; // [song] = votes
top7_t new_set;      // <votes, song>
top7_t top_set;      // <votes, song>

QueryType check_line(const std::string_view &line) {
  // unordered map would have the same performance with bigger memory usage
  static const std::map<QueryType, std::regex> e_type({
    { kVote,   std::regex("\\s*(0*[1-9]\\d{0,7}(\\s+|$))+") },
    { kNew,    std::regex("\\s*NEW\\s+0*[1-9]\\d{0,7}\\s*") },
    { kTop,    std::regex("\\s*TOP\\s*") },
    { kIgnore, std::regex("\\s*") }
  });

  for (const auto &[type, expr] : e_type)
    if (std::regex_match(line.begin(), line.end(), expr))
      return type;
  return kError;
}

void leave7(top7_t &s) {
  while (s.size() > 7)
    s.erase(s.begin());
}

bool is_valid(const std::vector<int> &songs) {
  uset_t duplicates;
  for (const auto &song : songs) {
    if (song > MAX || dropped.contains(song) || duplicates.contains(song))
      return false;
    duplicates.emplace(song);
  }
  return true;
}

template <QueryType qt> bool handle(const std::string &line);

template <> bool handle<kVote>(const std::string &line) {
  std::vector<int> songs;

  auto b = std::sregex_iterator(line.begin(), line.end(), e_num);
  auto e = std::sregex_iterator();

  for (std::sregex_iterator it = b; it != e; it++)
    songs.emplace_back(std::stoi(it->str()));

  if (!is_valid(songs))
    return false;

  for (const auto &song : songs) {
    new_set.erase(std::make_pair(votes_count[song], song));
    new_set.emplace(++votes_count[song], song);
    leave7(new_set);
  }

  return true;
}

template <> bool handle<kNew>(const std::string &line) {
  std::sregex_iterator b(line.begin(), line.end(), e_num);
  int new_max = std::stoi((++b)->str());

  if (new_max < MAX)
    return false;
  MAX = new_max;

  leave7(new_set);

  static umap_t prev_place_new; // [song] = place

  // sum up top7 for this voting
  int points = 7;
  for (auto it = new_set.rbegin(); it != new_set.rend(); it++, points--) {
    int song = it->second;
    std::cout << song << ' ';
    if (prev_place_new.contains(song))
      std::cout << points - prev_place_new[song] << std::endl;
    else
      std::cout << '-' << std::endl;

    prev_place_new.erase(song);
    top_set.erase(std::make_pair(points_count[song], song));
    points_count[song] += points;
    top_set.emplace(points_count[song], song);
  }

  leave7(top_set);

  for (const auto &[song, _] : prev_place_new)
    dropped.emplace(song);
  prev_place_new.clear();
  for (const auto &[_, song] : new_set)
    prev_place_new[song] = ++points;

  new_set.clear();
  votes_count.clear();

  return true;
}

template <> bool handle<kTop>([[maybe_unused]] const std::string &line) {
  static umap_t prev_place_top; // [song] = place
  int points = 7;
  for (auto it = top_set.rbegin(); it != top_set.rend(); it++, points--) {
    int song = it->second;
    std::cout << song << ' ';
    if (prev_place_top[song] != 0)
      std::cout << points - prev_place_top[song] << std::endl;
    else
      std::cout << '-' << std::endl;
  }

  prev_place_top.clear();
  for (const auto &[_, song] : top_set)
    prev_place_top[song] = ++points;

  return true;
}

} // namespace

int main() {
  std::string line;

  for (unsigned num = 1; getline(std::cin, line); num++) {
    if (QueryType qt = check_line(line); qt == kVote && handle<kVote>(line));
    else if (qt == kNew && handle<kNew>(line));
    else if (qt == kTop && handle<kTop>(line));
    else if (qt != kIgnore)
      std::cerr << "Error in line " << num << ": " << line << std::endl;
  }

  return 0;
}
