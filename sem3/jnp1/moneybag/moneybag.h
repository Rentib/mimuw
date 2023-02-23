#ifndef MONEYBAG_H_
#define MONEYBAG_H_

#include <cstdint>
#include <sstream>

class Moneybag {
public:
  using coin_number_t = uint64_t;

  Moneybag() = delete;
  constexpr Moneybag(coin_number_t _livre, coin_number_t _solidus,
                     coin_number_t _denier)
      : livre(_livre), solidus(_solidus), denier(_denier) {}
  constexpr Moneybag &operator=(const Moneybag &rhs) = default;
  constexpr Moneybag(const Moneybag &moneybag) = default;
  constexpr coin_number_t livre_number() const;
  constexpr coin_number_t solidus_number() const;
  constexpr coin_number_t denier_number() const;
  constexpr Moneybag operator+(const Moneybag &rhs) const;
  constexpr Moneybag operator-(const Moneybag &rhs) const;
  constexpr Moneybag &operator+=(const Moneybag &rhs);
  constexpr Moneybag &operator-=(const Moneybag &rhs);
  constexpr Moneybag operator*(coin_number_t rhs) const;
  constexpr Moneybag &operator*=(coin_number_t rhs);
  constexpr auto operator<=>(const Moneybag &rhs) const;
  constexpr bool operator==(const Moneybag &rhs) const = default;
  explicit constexpr operator bool() const;

private:
  coin_number_t livre, solidus, denier;
};

constexpr Moneybag Denier(0, 0, 1), Solidus(0, 1, 0), Livre(1, 0, 0);

constexpr Moneybag::coin_number_t Moneybag::livre_number() const {
  return livre;
}

constexpr Moneybag::coin_number_t Moneybag::solidus_number() const {
  return solidus;
}

constexpr Moneybag::coin_number_t Moneybag::denier_number() const {
  return denier;
}

constexpr Moneybag Moneybag::operator+(const Moneybag &rhs) const {
  if (livre > UINT64_MAX - rhs.livre_number() ||
      solidus > UINT64_MAX - rhs.solidus_number() ||
      denier > UINT64_MAX - rhs.denier_number())
    throw std::out_of_range(
        "a moneybag cannot have more than 2^64 - 1 coins of any type");

  return {livre + rhs.livre_number(), solidus + rhs.solidus_number(),
          denier + rhs.denier_number()};
}

constexpr Moneybag Moneybag::operator-(const Moneybag &rhs) const {
  if (livre < rhs.livre_number() || solidus < rhs.solidus_number() ||
      denier < rhs.denier_number())
    throw std::out_of_range("a moneybag cannot have negative amount of coins");

  return {livre - rhs.livre_number(), solidus - rhs.solidus_number(),
          denier - rhs.denier_number()};
}

constexpr Moneybag &Moneybag::operator+=(const Moneybag &rhs) {
  return *this = *this + rhs;
}

constexpr Moneybag &Moneybag::operator-=(const Moneybag &rhs) {
  return *this = *this - rhs;
}

constexpr Moneybag Moneybag::operator*(Moneybag::coin_number_t rhs) const {
  if (std::max(livre, std::max(solidus, denier)) > UINT64_MAX / rhs)
    throw std::out_of_range(
        "a moneybag cannot have more than 2^64 - 1 coins of any type");

  return {livre * rhs, solidus * rhs, denier * rhs};
}

constexpr Moneybag &Moneybag::operator*=(Moneybag::coin_number_t rhs) {
  return *this = *this * rhs;
}

constexpr Moneybag operator*(Moneybag::coin_number_t lhs, const Moneybag &rhs) {
  return rhs * lhs;
}

inline std::ostream &operator<<(std::ostream &os, const Moneybag &m) {
  return os << "(" << m.livre_number()
            << (m.livre_number() == 1 ? " livr" : " livres") << ", "
            << m.solidus_number()
            << (m.solidus_number() == 1 ? " solidus" : " soliduses") << ", "
            << m.denier_number()
            << (m.denier_number() == 1 ? " denier" : " deniers") << ")";
}

constexpr auto Moneybag::operator<=>(const Moneybag &rhs) const {
  unsigned smaller_count = 0, greater_count = 0;
  smaller_count = (livre < rhs.livre_number()) +
                  (solidus < rhs.solidus_number()) +
                  (denier < rhs.denier_number());
  greater_count = (livre > rhs.livre_number()) +
                  (solidus > rhs.solidus_number()) +
                  (denier > rhs.denier_number());
  if (smaller_count && greater_count)
    return std::partial_ordering::unordered;
  if (smaller_count)
    return std::partial_ordering::less;
  if (greater_count)
    return std::partial_ordering::greater;
  return std::partial_ordering::equivalent;
}

constexpr Moneybag::operator bool() const {
  return (livre | solidus | denier) != 0; // | instead of + to avoid overflow
}

class Value {
public:
  constexpr Value() : upper(0), lower(0) {}
  constexpr Value(const Moneybag &m);
  constexpr Value(Moneybag::coin_number_t denier) : upper(0), lower(denier) {}
  constexpr Value(const Value &value) = default;
  constexpr Value &operator=(const Value &rhs) = default;
  constexpr auto operator<=>(const Value &rhs) const = default;
  constexpr auto operator<=>(Moneybag::coin_number_t denier) const;
  explicit operator std::string() const;

private:
  Moneybag::coin_number_t upper, lower; // upper 64 and lower 64 bits
};

constexpr Value::Value(const Moneybag &m) {
  __uint128_t tmp = (__uint128_t)m.livre_number() * 240 +
                    (__uint128_t)m.solidus_number() * 12 +
                    (__uint128_t)m.denier_number();
  upper = tmp >> 64;
  lower = tmp & (Moneybag::coin_number_t)~0;
}

constexpr auto Value::operator<=>(Moneybag::coin_number_t _denier) const {
  if (!upper && lower < _denier)
    return -1;
  if (upper || lower > _denier)
    return 1;
  return 0;
}

inline Value::operator std::string() const {
  // This method of printing uses the fact that Value is smaller than 1e36
  // x = floor(x / 1e18) + (x mod 1e18)
  // In our case x / 1e18 < 2^65 so we can use std::to_string().
  // Modulo cuts leading zeros so we need to insert them manually.
  if (!upper)
    return std::to_string(lower);
  __uint128_t tmp = ((__uint128_t)upper << 64) | lower;
  uint64_t l = tmp / 1e18, r = tmp % (uint64_t)1e18;
  std::string a = std::to_string(l), b = std::to_string(r);
  return a + std::string(18 - b.size(), '0') + b;
}

#endif // MONEYBAG_H_
