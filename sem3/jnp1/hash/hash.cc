#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hash.h"

namespace jnp1 {

namespace {

using sequence_t = std::vector<uint64_t>;
using ulong = unsigned long;

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;

std::ostream &operator<<(std::ostream &os,
                         const std::pair<const uint64_t *, size_t> &&p) {
  if (p.first == nullptr)
    return os << "NULL";
  os << "\"";
  for (unsigned i = 0; i < p.second; i++)
    os << p.first[i] << (i < p.second - 1 ? " " : "");
  return os << "\"";
}

std::ostream &operator<<(std::ostream &os, const sequence_t &seq) {
  return os << std::make_pair(seq.data(), seq.size());
}
#endif

template <typename... Args> inline void log(const char *func, Args &&...args) {
  if constexpr (debug) {
    static std::ios_base::Init init;
    ((std::cerr << func) << ... << std::forward<Args>(args)) << std::endl;
  }
}

sequence_t make_sequence_t(const uint64_t *seq, size_t size) {
  sequence_t res;
  for (unsigned i = 0; i < size; i++)
    res.emplace_back(seq[i]);
  return res;
}

struct Hash_s {
  hash_function_t hash_function;
  Hash_s(hash_function_t func) : hash_function(func) {}
  size_t operator()(const sequence_t &seq) const noexcept {
    return (*hash_function)(seq.data(), seq.size());
  }
};

auto &mp() {
  static std::unordered_map<ulong, std::unordered_set<sequence_t, Hash_s>> m;
  return m;
}

bool check_args(const char *func, ulong id, const uint64_t *seq, size_t size) {
  bool err = false;

  if (seq == nullptr) {
    log(func, ": invalid pointer (NULL)");
    err = true;
  }
  if (!size) {
    log(func, ": invalid size (0)");
    err = true;
  }

  if (!err && !mp().count(id)) {
    log(func, ": hash table #", id, " does not exist");
    err = true;
  }

  return !err;
}

} // namespace

ulong hash_create(hash_function_t hash_function) {
  static ulong id = 0;
  log(__func__, '(', &hash_function, ')');
  auto tmp = std::unordered_set<sequence_t, Hash_s>({}, Hash_s(hash_function));
  mp().emplace(id, tmp);
  log(__func__, ": hash table #", id, " created");
  return id++;
}

void hash_delete(ulong id) {
  log(__func__, '(', id, ')');
  if (mp().count(id)) {
    mp().erase(id);
    log(__func__, ": hash table #", id, " deleted");
  } else {
    log(__func__, ": hash table #", id, " does not exist");
  }
}

size_t hash_size(ulong id) {
  log(__func__, '(', id, ')');
  if (mp().count(id)) {
    size_t sz = mp().at(id).size();
    log(__func__, ": hash table #", id, " contains ", sz, " element(s)");
    return sz;
  } else {
    log(__func__, ": hash table #", id, " does not exist");
    return 0;
  }
}

bool hash_insert(ulong id, const uint64_t *seq, size_t size) {
  log(__func__, '(', id, ", ", std::make_pair(seq, size), ", ", size, ')');

  if (!check_args(__FUNCTION__, id, seq, size))
    return false;

  const sequence_t s = make_sequence_t(seq, size);
  if (mp().at(id).count(s)) {
    log(__func__, ": hash table #", id, ", sequence ", s, " was present");
    return false;
  }

  mp().at(id).emplace(s);
  log(__func__, ": hash table #", id, ", sequence ", s, " inserted");
  return true;
}

bool hash_remove(ulong id, const uint64_t *seq, size_t size) {
  log(__func__, '(', id, ", ", std::make_pair(seq, size), ", ", size, ')');

  if (!check_args(__FUNCTION__, id, seq, size))
    return false;

  const sequence_t s = make_sequence_t(seq, size);
  if (!mp().at(id).count(s)) {
    log(__func__, ": hash table #", id, ", sequence ", s, " was not present");
    return false;
  }

  mp().at(id).erase(s);
  log(__func__, ": hash table #", id, ", sequence ", s, " removed");
  return true;
}

void hash_clear(ulong id) {
  log(__func__, '(', id, ')');
  if (!mp().count(id)) {
    log(__func__, ": hash table #", id, " does not exist");
  } else if (!mp().at(id).size()) {
    log(__func__, ": hash table #", id, " was empty");
  } else {
    mp().at(id).clear();
    log(__func__, ": hash table #", id, " cleared");
  }
}

bool hash_test(ulong id, const uint64_t *seq, size_t size) {
  log(__func__, '(', id, ", ", std::make_pair(seq, size), ", ", size, ')');

  if (!check_args(__func__, id, seq, size))
    return false;

  const sequence_t s = make_sequence_t(seq, size);
  const bool res = mp().at(id).count(s) > 0;
  log(__func__, ": hash table #", id, ", sequence ", s, " is",
      res ? "" : " not", " present");
  return res;
}

} // namespace jnp1
