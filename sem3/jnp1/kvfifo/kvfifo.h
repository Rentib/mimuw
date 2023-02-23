#ifndef KVFIFO_H
#define KVFIFO_H

#include <list>
#include <map>
#include <memory>
#include <stdexcept>

template <typename K, typename V> class kvfifo {
private:
  using list_ptr_t = typename std::list<std::pair<K, V>>::iterator;
  using map_t = std::map<K, std::list<list_ptr_t>>;
  using list_t = std::list<std::pair<K, V>>;

  // map of lists of pointers to values of the same key
  std::shared_ptr<map_t> kv_map;
  // list of pairs <Key, Value>
  std::shared_ptr<list_t> kv_list;
  bool must_copy;

  inline void copy() {
    if (must_copy || !kv_map.unique() || !kv_list.unique()) {
      try {
        kvfifo new_this{};
        for (const auto &[key, val] : *kv_list)
          new_this.push(key, val);
        *this = new_this;
      } catch (...) {
        throw;
      }
    }
  }

public:
  class k_iterator {
  private:
    typename map_t::const_iterator it;

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = const K;
    using difference_type = ptrdiff_t;
    using pointer = const K *;
    using reference = const K &;

    inline k_iterator() = default;
    inline k_iterator(K it) : it(it) {}
    inline k_iterator(const k_iterator &other) : it(other.it) {}
    inline k_iterator(typename map_t::const_iterator &&it) : it(it) {}

    inline k_iterator &operator++() noexcept {
      ++it;
      return *this;
    }

    inline k_iterator operator++(int) noexcept {
      auto prev = *this;
      ++*this;
      return prev;
    }

    inline k_iterator &operator--() noexcept {
      --it;
      return *this;
    }

    inline k_iterator operator--(int) noexcept {
      auto prev = *this;
      --*this;
      return prev;
    }

    inline bool operator==(const k_iterator &other) const noexcept {
      return it == other.it;
    }

    inline bool operator!=(const k_iterator &other) const noexcept {
      return !this->operator==(other);
    }

    inline k_iterator &operator=(const k_iterator &other) noexcept = default;

    inline reference operator*() const noexcept { return (*it).first; }
    inline pointer operator->() const noexcept { return &(it->first); }
  };

  inline kvfifo()
      : kv_map(std::make_shared<map_t>()), kv_list(std::make_shared<list_t>()),
        must_copy(false) {}
  inline kvfifo(const kvfifo &other)
      : kv_map(other.kv_map), kv_list(other.kv_list), must_copy(other.must_copy) {
    try {
      if (must_copy)
        copy();
    } catch (...) {
      throw;
    }
  }
  inline kvfifo(kvfifo &&other) : must_copy(other.must_copy) {
    kv_map = other.kv_map;
    kv_list = other.kv_list;
    other.clear();

    try {
      if (must_copy)
        copy();
    } catch (...) {
      throw;
    }
  };

  inline kvfifo &operator=(kvfifo other) {
    kv_map.swap(other.kv_map);
    kv_list.swap(other.kv_list);
    must_copy = other.must_copy;

    try {
      if (must_copy)
        copy();
    } catch (...) {
      throw;
    }

    return *this;
  };

  inline void push(const K &key, const V &val) {
    try {
      copy();
    } catch (...) {
      throw;
    }

    bool do_pop_back = false;

    try {
      kv_list->emplace_back(key, val);
      do_pop_back = true;
      (*kv_map)[key].emplace_back(std::prev(kv_list->end()));
    } catch (...) {
      if (do_pop_back) {
        kv_list->pop_back();
        if (kv_map->contains(key) && (*kv_map)[key].empty())
          kv_map->erase(key);
      }
      throw;
    }
  }

  inline void pop() {
    if (kv_list->empty())
      throw std::invalid_argument("kvfifo: empty");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto key = kv_list->front().first;
    kv_list->pop_front();

    auto &bucket = (*kv_map)[key];
    bucket.pop_front();
    if (bucket.empty())
      kv_map->erase(key);
  }

  inline void pop(const K &key) {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &bucket = (*kv_map)[key];
    kv_list->erase(bucket.front());
    bucket.pop_front();
  }

  inline void move_to_back(const K &key) {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &bucket = (*kv_map)[key];
    for (auto it : bucket)
      kv_list->splice(kv_list->end(), *kv_list, it);
  }

  inline std::pair<const K &, V &> front() {
    if (kv_list->empty())
      throw std::invalid_argument("kvfifo: empty");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &[key, val] = kv_list->front();
    must_copy = true;
    return {key, val};
  }

  inline std::pair<const K &, const V &> front() const {
    if (kv_list->empty())
      throw std::invalid_argument("kvfifo: empty");

    auto &[key, val] = kv_list->front();
    return {key, val};
  }
  inline std::pair<const K &, V &> back() {
    if (kv_list->empty())
      throw std::invalid_argument("kvfifo: empty");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &[key, val] = kv_list->back();
    must_copy = true;
    return {key, val};
  }

  inline std::pair<const K &, const V &> back() const {
    if (kv_list->empty())
      throw std::invalid_argument("kvfifo: empty");

    auto &[key, val] = kv_list->back();
    return {key, val};
  }

  inline std::pair<const K &, V &> first(const K &key) {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &it = (*kv_map)[key].front();
    auto &val = it->second;
    must_copy = true;
    return {key, val};
  }

  inline std::pair<const K &, const V &> first(const K &key) const {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    auto &it = kv_map->find(key)->second.front();
    auto &val = it->second;
    return {key, val};
  }

  inline std::pair<const K &, V &> last(const K &key) {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    try {
      copy();
    } catch (...) {
      throw;
    }

    auto &it = (*kv_map)[key].back();
    auto &val = it->second;
    must_copy = true;
    return {key, val};
  }

  inline std::pair<const K &, const V &> last(const K &key) const {
    if (!kv_map->contains(key))
      throw std::invalid_argument("kvfifo: key not found");

    auto &it = kv_map->find(key)->second.back();
    auto &val = it->second;
    return {key, val};
  }

  inline size_t size() const noexcept { return kv_list->size(); }

  inline bool empty() const noexcept { return kv_list->empty(); }

  inline size_t count(const K &key) const noexcept {
    return kv_map->contains(key) ? kv_map->find(key)->second.size() : 0;
  };

  inline void clear() {
    try {
      copy();
      kv_map->clear();
      kv_list->clear();
    } catch (...) {
      throw;
    }
  }

  inline k_iterator k_begin() const noexcept { return {kv_map->begin()}; }
  inline k_iterator k_end() const noexcept { return {kv_map->end()}; }
};

#endif // KVFIFO_H
